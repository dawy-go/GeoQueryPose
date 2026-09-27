from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.amp_utils import autocast_disabled
from utils.keypoint_utils import solve_weighted_rotation_from_points


class JointCategoryHead(nn.Module):
    """Predict calibrated class logits as a residual over detector labels."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        prior_logit: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.prior_logit = float(prior_logit)
        self.detector_embedding = nn.Embedding(self.num_classes, feature_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim * 2 + 1),
            nn.Linear(feature_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        global_feature: torch.Tensor,
        detector_category: torch.Tensor,
        detector_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        detector_category = detector_category.reshape(-1).long()
        if detector_score is None:
            detector_score = global_feature.new_ones(global_feature.shape[0])
        detector_score = detector_score.to(global_feature).reshape(-1, 1)
        feature = torch.cat(
            [
                global_feature,
                self.detector_embedding(detector_category),
                detector_score,
            ],
            dim=1,
        )
        residual = self.network(feature)
        prior = residual.new_zeros(residual.shape)
        prior.scatter_(1, detector_category[:, None], self.prior_logit)
        return prior + residual


class JointPointNOCSHead(nn.Module):
    """Predict canonical coordinates and correspondence confidence per point."""

    def __init__(
        self,
        global_dim: int,
        point_dim: int = 9,
        hidden_dim: int = 256,
        global_proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, global_proj_dim),
            nn.SiLU(),
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim + global_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.global_proj(global_feature).unsqueeze(1)
        projected = projected.expand(-1, point_features.shape[1], -1)
        output = self.point_mlp(torch.cat([point_features, projected], dim=-1))
        return 0.6 * torch.tanh(output[..., :3]), output[..., 3]


def solve_nocs_geometry(
    pred_nocs: torch.Tensor,
    confidence_logits: torch.Tensor,
    pred_size: torch.Tensor,
    camera_points: torch.Tensor,
) -> dict[str, torch.Tensor]:
    confidence = torch.sigmoid(confidence_logits).clamp_min(1.0e-4)
    metric_points = pred_nocs * pred_size.unsqueeze(1).clamp_min(1.0e-4)
    rotation, translation, residual, valid_ratio = solve_weighted_rotation_from_points(
        metric_points,
        camera_points,
        confidence,
        svd_regularization=1.0e-4,
    )
    finite = (
        torch.isfinite(rotation).flatten(1).all(dim=1)
        & torch.isfinite(translation).all(dim=1)
        & torch.isfinite(residual)
    )
    return {
        "rotation": rotation,
        "translation": translation,
        "residual": residual,
        "valid_ratio": valid_ratio,
        "valid": finite & (valid_ratio > 0.0),
        "confidence": confidence,
        "metric_points": metric_points,
    }


class GeometryResidualHead(nn.Module):
    """Identity-initialized full XYZ/size correction with a geometry gate."""

    numeric_dim = 12

    def __init__(
        self,
        query_dim: int,
        num_classes: int,
        category_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        max_translation_ratio: float = 0.5,
        max_log_size_residual: float = 0.35,
        initial_gate_logit: float = -8.0,
        rotation_gate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_translation_ratio = float(max_translation_ratio)
        self.max_log_size_residual = float(max_log_size_residual)
        self.rotation_gate_scale = float(rotation_gate_scale)
        if not 0.0 <= self.rotation_gate_scale <= 1.0:
            raise ValueError("rotation_gate_scale must lie in [0, 1]")
        self.category_embedding = nn.Embedding(num_classes, category_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(query_dim + category_dim + self.numeric_dim),
            nn.Linear(query_dim + category_dim + self.numeric_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[6] = float(initial_gate_logit)

    def forward(
        self,
        query: torch.Tensor,
        category: torch.Tensor,
        base_translation: torch.Tensor,
        base_rotation: torch.Tensor,
        base_size: torch.Tensor,
        geometry_translation: torch.Tensor,
        geometry_rotation: torch.Tensor,
        geometry_residual: torch.Tensor,
        geometry_valid_ratio: torch.Tensor,
        geometry_valid: torch.Tensor,
        class_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        category = category.reshape(-1).long()
        base_size = base_size.clamp_min(1.0e-4)
        size_scale = base_size.mean(dim=1, keepdim=True).clamp_min(1.0e-4)
        normalized_geometry_delta = (
            geometry_translation - base_translation
        ) / size_scale
        numeric = torch.cat(
            [
                base_translation,
                torch.log(base_size),
                normalized_geometry_delta,
                (geometry_residual[:, None] / size_scale),
                geometry_valid_ratio[:, None],
                class_probability[:, None],
            ],
            dim=1,
        )
        raw = self.network(
            torch.cat([query, self.category_embedding(category), numeric], dim=1)
        )
        translation_delta = (
            self.max_translation_ratio
            * torch.tanh(raw[:, :3])
            * size_scale
        )
        log_size_delta = self.max_log_size_residual * torch.tanh(raw[:, 3:6])
        gate = torch.sigmoid(raw[:, 6])
        gate = torch.where(geometry_valid, gate, torch.zeros_like(gate))
        rotation_gate = (gate * self.rotation_gate_scale).clamp(0.0, 1.0)
        safe_geometry_rotation = torch.where(
            geometry_valid[:, None, None], geometry_rotation, base_rotation
        )
        mixed_rotation = (
            (1.0 - rotation_gate[:, None, None]) * base_rotation
            + rotation_gate[:, None, None] * safe_geometry_rotation
        )
        with autocast_disabled(mixed_rotation.device):
            mixed_rotation_fp32 = mixed_rotation.float()
            first = F.normalize(
                mixed_rotation_fp32[:, :, 0], dim=1, eps=1.0e-6
            )
            second_raw = mixed_rotation_fp32[:, :, 1]
            second_raw = second_raw - (
                first * second_raw
            ).sum(dim=1, keepdim=True) * first
            second = F.normalize(second_raw, dim=1, eps=1.0e-6)
            third = torch.linalg.cross(first, second, dim=1)
            refined_rotation = torch.stack(
                (first, second, third), dim=2
            ).to(base_rotation.dtype)
        refined_translation = (
            base_translation
            + gate[:, None] * (geometry_translation - base_translation)
            + translation_delta
        )
        refined_size = base_size * torch.exp(log_size_delta)
        return {
            "translation": refined_translation,
            "rotation": refined_rotation,
            "size": refined_size,
            "translation_delta": translation_delta,
            "log_size_delta": log_size_delta,
            "gate": gate,
            "quality_logit": raw[:, 7],
        }
