from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


CENTER_ANCHOR_FEATURE_DIM = 18


def mask_center_from_crop(
        mask_crop: np.ndarray,
        bbox_xyxy: Sequence[float],
) -> Tuple[np.ndarray, bool, float]:
    """Return the full-image foreground centroid and crop occupancy ratio."""
    mask = np.asarray(mask_crop) > 0
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    bbox_center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return bbox_center, False, 0.0
    center = np.asarray([x1 + cols.mean(), y1 + rows.mean()], dtype=np.float32)
    return center, True, float(rows.size / max(mask.size, 1))


def project_translation_to_image(translation: torch.Tensor, cam_k: torch.Tensor) -> torch.Tensor:
    """Project camera-space XYZ translations to full-image pixel coordinates."""
    fx, fy, cx, cy = cam_k.unbind(dim=-1)
    z = translation[:, 2].clamp_min(1.0e-6)
    u = fx * translation[:, 0] / z + cx
    v = fy * translation[:, 1] / z + cy
    return torch.stack([u, v], dim=-1)


def backproject_image_center(uv: torch.Tensor, z: torch.Tensor, cam_k: torch.Tensor) -> torch.Tensor:
    """Back-project pixels at the supplied depth without changing that depth."""
    fx, fy, cx, cy = cam_k.unbind(dim=-1)
    safe_z = torch.nan_to_num(z, nan=1.0e-3, posinf=1.0e-3, neginf=1.0e-3).clamp_min(1.0e-6)
    x = (uv[:, 0] - cx) * safe_z / fx.clamp_min(1.0e-6)
    y = (uv[:, 1] - cy) * safe_z / fy.clamp_min(1.0e-6)
    return torch.stack([x, y, safe_z], dim=-1)


class CenterAnchorGate(nn.Module):
    """Blend point, bbox, and mask image rays while preserving point depth."""

    def __init__(
            self,
            pose_dim: int,
            hidden_dim: int = 128,
            point_init_logit: float = 5.0,
            candidate_dropout: float = 0.0,
    ):
        super().__init__()
        self.candidate_dropout = max(0.0, min(1.0, float(candidate_dropout)))
        self.gate = nn.Sequential(
            nn.LayerNorm(CENTER_ANCHOR_FEATURE_DIM),
            nn.Linear(CENTER_ANCHOR_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.translation_context = nn.Sequential(
            nn.LayerNorm(CENTER_ANCHOR_FEATURE_DIM + 3),
            nn.Linear(CENTER_ANCHOR_FEATURE_DIM + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(
                torch.tensor([float(point_init_logit), 0.0, 0.0], dtype=self.gate[-1].bias.dtype)
            )
        # The translation denoiser initially behaves exactly as before; this
        # adapter learns its contribution gradually.
        nn.init.zeros_(self.translation_context[-1].weight)
        nn.init.zeros_(self.translation_context[-1].bias)

    def forward(
            self,
            point_anchor: torch.Tensor,
            points: torch.Tensor,
            point_valid: torch.Tensor,
            boxes_yxyx: torch.Tensor,
            cam_k: torch.Tensor,
            image_hw: torch.Tensor,
            mask_center_uv: Optional[torch.Tensor] = None,
            mask_center_valid: Optional[torch.Tensor] = None,
            mask_area_ratio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        dtype = point_anchor.dtype
        device = point_anchor.device
        boxes = boxes_yxyx.to(device=device, dtype=dtype).reshape(-1, 4)
        intrinsics = cam_k.to(device=device, dtype=dtype).reshape(-1, 4)
        image_hw = image_hw.to(device=device, dtype=dtype).reshape(-1, 2)

        y1, x1, y2, x2 = boxes.unbind(dim=-1)
        bbox_wh = torch.stack([(x2 - x1).abs(), (y2 - y1).abs()], dim=-1).clamp_min(1.0)
        bbox_uv = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dim=-1)

        if mask_center_uv is None:
            mask_uv = bbox_uv
        else:
            mask_uv = mask_center_uv.to(device=device, dtype=dtype).reshape(-1, 2)
        if mask_center_valid is None:
            mask_valid = torch.zeros((point_anchor.shape[0],), device=device, dtype=torch.bool)
        else:
            mask_valid = mask_center_valid.to(device=device).reshape(-1).bool()
        mask_uv = torch.where(mask_valid.unsqueeze(-1), mask_uv, bbox_uv)
        if mask_area_ratio is None:
            mask_ratio = torch.zeros((point_anchor.shape[0],), device=device, dtype=dtype)
        else:
            mask_ratio = mask_area_ratio.to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0)

        point_uv = project_translation_to_image(point_anchor, intrinsics)
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
        focal = torch.stack([fx, fy], dim=-1).clamp_min(1.0e-6)
        principal = torch.stack([cx, cy], dim=-1)
        ray_features = torch.cat([
            (point_uv - principal) / focal,
            (bbox_uv - principal) / focal,
            (mask_uv - principal) / focal,
        ], dim=-1)
        delta_features = torch.cat([
            (point_uv - bbox_uv) / bbox_wh,
            (point_uv - mask_uv) / bbox_wh,
        ], dim=-1)

        image_h = image_hw[:, 0].clamp_min(1.0)
        image_w = image_hw[:, 1].clamp_min(1.0)
        bbox_features = torch.stack([
            bbox_wh[:, 0] / image_w,
            bbox_wh[:, 1] / image_h,
            torch.log(bbox_wh[:, 0] / bbox_wh[:, 1]),
        ], dim=-1)

        valid = point_valid.to(device=device).bool()
        valid_f = valid.to(dtype)
        valid_count = valid_f.sum(dim=1).clamp_min(1.0)
        z = torch.nan_to_num(points[:, :, 2].to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        z_mean = (z * valid_f).sum(dim=1) / valid_count
        z_var = (((z - z_mean.unsqueeze(1)) ** 2) * valid_f).sum(dim=1) / valid_count
        z_rel_std = torch.sqrt(z_var.clamp_min(0.0)) / z_mean.abs().clamp_min(1.0e-6)
        point_valid_ratio = valid_f.mean(dim=1)
        border = (
            (x1 <= 0.0) | (y1 <= 0.0)
            | (x2 >= image_w - 1.0) | (y2 >= image_h - 1.0)
        ).to(dtype)

        quality_features = torch.stack([
            mask_ratio,
            point_valid_ratio,
            z_rel_std,
            mask_valid.to(dtype),
            border,
        ], dim=-1)
        features = torch.cat([ray_features, delta_features, bbox_features, quality_features], dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        logits = self.gate(features)
        candidate_valid = torch.stack([
            torch.ones_like(mask_valid),
            torch.ones_like(mask_valid),
            mask_valid,
        ], dim=-1)
        if self.training and self.candidate_dropout > 0.0:
            keep_optional = torch.rand_like(logits[:, 1:]) >= self.candidate_dropout
            candidate_valid[:, 1:] = candidate_valid[:, 1:] & keep_optional
        logits = logits.masked_fill(~candidate_valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)

        candidate_uv = torch.stack([point_uv, bbox_uv, mask_uv], dim=1)
        fused_uv = (candidate_uv * weights.unsqueeze(-1)).sum(dim=1)
        coarse_translation = backproject_image_center(fused_uv, point_anchor[:, 2], intrinsics)
        context = self.translation_context(torch.cat([features, weights], dim=-1)).unsqueeze(1)

        return {
            "coarse_translation": coarse_translation,
            "point_anchor": point_anchor,
            "point_center_uv": point_uv,
            "bbox_center_uv": bbox_uv,
            "mask_center_uv": mask_uv,
            "fused_center_uv": fused_uv,
            "anchor_gate_weights": weights,
            "anchor_geometry_features": features,
            "anchor_context": context,
            "anchor_cam_k": intrinsics,
            "anchor_bbox_wh": bbox_wh,
            "anchor_mask_valid": mask_valid,
        }
