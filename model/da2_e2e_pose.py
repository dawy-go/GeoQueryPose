from __future__ import annotations

import math
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from modules.da2_pose_modules import DA2VisualEncoder
from utils.rotation_utils import six_d_to_rotation_matrix


def _select(cfg: Any, key: str, default: Any) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1.0e-6)
    return math.log(math.expm1(value))


class ProjectiveQueryLayer(nn.Module):
    """One object-query update over differentiably lifted RGB tokens."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=query,
            key=context,
            value=context,
            need_weights=False,
        )
        query = self.attention_norm(query + attended)
        return self.ffn_norm(query + self.ffn(query))


class ResidualFeatureAdapter(nn.Module):
    """Branch-specific adaptation with an exact identity initialization."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.residual(feature)


class DA2E2EPose(nn.Module):
    """RGB-crop to metric category-level pose with differentiable lifting.

    Detection boxes and categories remain externally supplied in this first
    controlled stage.  Offline depth and point clouds are never consumed by
    the forward pass.  A DA2 RGB backbone predicts metric depth tokens, which
    are unprojected with the camera intrinsics before an object query predicts
    rotation, projective centre/depth, and metric size.
    """

    requires_octree = False

    def __init__(
        self,
        cfg: Any,
        visual_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_dim = int(_select(cfg, "da2.token_dim", 256))
        self.num_classes = int(_select(cfg, "e2e_pose.num_classes", 6))
        self.depth_min = float(_select(cfg, "e2e_pose.depth_min_m", 0.10))
        self.depth_max = float(_select(cfg, "e2e_pose.depth_max_m", 3.00))
        self.depth_init = float(_select(cfg, "e2e_pose.depth_init_m", 0.70))
        self.size_min = float(_select(cfg, "e2e_pose.size_min_m", 1.0e-4))
        self.size_init = float(_select(cfg, "e2e_pose.size_init_m", 0.20))
        self.max_uv_offset_ratio = float(
            _select(cfg, "e2e_pose.max_uv_offset_ratio", 0.25)
        )
        self.max_log_z_residual = float(
            _select(cfg, "e2e_pose.max_log_z_residual", 0.70)
        )
        self.decouple_translation_size = bool(
            _select(cfg, "e2e_pose.decouple_translation_size", False)
        )
        self.detach_pose_query_for_ts = bool(
            _select(cfg, "e2e_pose.detach_pose_query_for_ts", False)
        )
        if not 0.0 < self.depth_min < self.depth_max:
            raise ValueError("e2e_pose depth range must satisfy 0 < min < max")
        if not self.depth_min < self.depth_init < self.depth_max:
            raise ValueError("e2e_pose.depth_init_m must lie inside the depth range")

        if visual_encoder is None:
            visual_encoder = DA2VisualEncoder(
                model_name=str(
                    _select(
                        cfg,
                        "da2.model_name",
                        "depth-anything/Depth-Anything-V2-Small-hf",
                    )
                ),
                input_size=int(_select(cfg, "da2.input_size", 224)),
                patch_size=int(_select(cfg, "da2.patch_size", 14)),
                token_dim=self.token_dim,
                token_grid=int(_select(cfg, "da2.token_grid", 8)),
                num_scales=int(_select(cfg, "da2.num_scales", 4)),
                num_heads=int(_select(cfg, "da2.num_heads", 8)),
                freeze_backbone=bool(_select(cfg, "da2.freeze_backbone", True)),
                local_files_only=bool(_select(cfg, "da2.local_files_only", False)),
            )
        self.visual_encoder = visual_encoder

        hidden_dim = int(_select(cfg, "e2e_pose.depth_hidden_dim", self.token_dim))
        self.depth_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.depth_confidence_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
        )
        self.geometry_embed = nn.Sequential(
            nn.Linear(6, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.context_norm = nn.LayerNorm(self.token_dim)
        self.category_embedding = nn.Embedding(self.num_classes, self.token_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(self.token_dim * 2, self.token_dim),
            nn.LayerNorm(self.token_dim),
        )
        self.decoder_layers = nn.ModuleList(
            ProjectiveQueryLayer(
                dim=self.token_dim,
                num_heads=int(_select(cfg, "e2e_pose.num_heads", 8)),
                dropout=float(_select(cfg, "e2e_pose.dropout", 0.10)),
            )
            for _ in range(int(_select(cfg, "e2e_pose.decoder_layers", 4)))
        )

        self.center_offset_head = nn.Linear(self.token_dim, 2)
        self.log_z_residual_head = nn.Linear(self.token_dim, 1)
        self.rotation_head = nn.Linear(self.token_dim, 6)
        self.size_head = nn.Linear(self.token_dim, 3)
        if self.decouple_translation_size:
            branch_dropout = float(
                _select(cfg, "e2e_pose.branch_adapter_dropout", 0.10)
            )
            self.translation_feature_adapter = ResidualFeatureAdapter(
                self.token_dim, branch_dropout
            )
            self.size_feature_adapter = ResidualFeatureAdapter(
                self.token_dim, branch_dropout
            )
        self._initialize_output_heads()

    def _initialize_output_heads(self) -> None:
        depth_fraction = (
            (self.depth_init - self.depth_min)
            / (self.depth_max - self.depth_min)
        )
        nn.init.normal_(self.depth_head[-1].weight, std=1.0e-3)
        nn.init.constant_(self.depth_head[-1].bias, _inverse_sigmoid(depth_fraction))
        nn.init.zeros_(self.depth_confidence_head[-1].weight)
        nn.init.zeros_(self.depth_confidence_head[-1].bias)

        nn.init.normal_(self.center_offset_head.weight, std=1.0e-3)
        nn.init.zeros_(self.center_offset_head.bias)
        nn.init.normal_(self.log_z_residual_head.weight, std=1.0e-3)
        nn.init.zeros_(self.log_z_residual_head.bias)

        nn.init.normal_(self.rotation_head.weight, std=1.0e-3)
        with torch.no_grad():
            self.rotation_head.bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )
        nn.init.normal_(self.size_head.weight, std=1.0e-3)
        nn.init.constant_(
            self.size_head.bias,
            _inverse_softplus(max(self.size_init - self.size_min, 1.0e-4)),
        )

    def setup_data_parallel(self, gpu_ids) -> None:
        # The current training launcher owns device placement.  This route has
        # no OCNN objects and can later be wrapped by DDP without model changes.
        del gpu_ids

    @staticmethod
    def _category_indices(inputs: dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        category = inputs.get("category_label")
        if category is None:
            raise KeyError("DA2E2EPose requires category_label")
        category = category.reshape(batch_size, -1)[:, 0].long()
        return category

    @staticmethod
    def _bbox(inputs: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        bbox = inputs.get("pred_bboxes", inputs.get("bbox"))
        if bbox is None:
            raise KeyError("DA2E2EPose requires bbox or pred_bboxes")
        bbox = bbox.to(dtype=dtype).reshape(-1, 4)
        y1, x1, y2, x2 = bbox.unbind(dim=1)
        x2 = torch.maximum(x2, x1 + 1.0)
        y2 = torch.maximum(y2, y1 + 1.0)
        return torch.stack([y1, x1, y2, x2], dim=1)

    def _intrinsics(
        self,
        inputs: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        intrinsics = inputs.get("cam_k")
        if intrinsics is None:
            values = _select(
                self.cfg,
                "e2e_pose.intrinsics",
                [591.0125, 590.16775, 322.525, 244.11084],
            )
            intrinsics = torch.as_tensor(values, device=device, dtype=dtype)
        else:
            intrinsics = intrinsics.to(device=device, dtype=dtype)
        if intrinsics.ndim == 1:
            intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1)
        elif intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
            intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
        if intrinsics.shape[-2:] == (3, 3):
            fx = intrinsics[:, 0, 0]
            fy = intrinsics[:, 1, 1]
            cx = intrinsics[:, 0, 2]
            cy = intrinsics[:, 1, 2]
        else:
            intrinsics = intrinsics.reshape(batch_size, -1)
            if intrinsics.shape[1] != 4:
                raise ValueError("cam_k must be [B,4] or [B,3,3]")
            fx, fy, cx, cy = intrinsics.unbind(dim=1)
        return (
            fx.clamp_min(1.0e-6),
            fy.clamp_min(1.0e-6),
            cx,
            cy,
        )

    @staticmethod
    def _token_grid(token_count: int, device: torch.device, dtype: torch.dtype):
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError(
                f"DA2E2EPose requires a square token grid, got {token_count} tokens"
            )
        rows = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
        cols = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
        grid_v, grid_u = torch.meshgrid(rows, cols, indexing="ij")
        return grid_u.reshape(1, -1), grid_v.reshape(1, -1), side

    def _lift_tokens(
        self,
        tokens: torch.Tensor,
        depth_tokens: torch.Tensor,
        bbox: torch.Tensor,
        intrinsics: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size, token_count, _ = tokens.shape
        grid_u, grid_v, side = self._token_grid(
            token_count,
            tokens.device,
            tokens.dtype,
        )
        y1, x1, y2, x2 = bbox.unbind(dim=1)
        width = (x2 - x1).clamp_min(1.0)
        height = (y2 - y1).clamp_min(1.0)
        pixel_u = x1[:, None] + grid_u * width[:, None]
        pixel_v = y1[:, None] + grid_v * height[:, None]
        fx, fy, cx, cy = intrinsics
        ray_x = (pixel_u - cx[:, None]) / fx[:, None]
        ray_y = (pixel_v - cy[:, None]) / fy[:, None]
        xyz = torch.stack(
            [ray_x * depth_tokens, ray_y * depth_tokens, depth_tokens],
            dim=-1,
        )
        uv_normalized = torch.stack(
            [grid_u.expand(batch_size, -1) * 2.0 - 1.0,
             grid_v.expand(batch_size, -1) * 2.0 - 1.0],
            dim=-1,
        )
        geometry = torch.cat(
            [xyz, uv_normalized, torch.log(depth_tokens.clamp_min(1.0e-6)).unsqueeze(-1)],
            dim=-1,
        )
        return geometry, pixel_u, pixel_v, side

    def encode_pose_features(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Encode an object crop once so lightweight category hypotheses can share it."""
        rgb = inputs.get("rgb")
        if rgb is None:
            raise KeyError("DA2E2EPose requires rgb")
        batch_size = rgb.shape[0]
        visual = self.visual_encoder(rgb)
        tokens = visual["tokens"]
        global_feature = visual["global"]

        depth_logits = self.depth_head(tokens).squeeze(-1)
        depth_tokens = self.depth_min + (
            self.depth_max - self.depth_min
        ) * torch.sigmoid(depth_logits)
        confidence = torch.softmax(
            self.depth_confidence_head(tokens).squeeze(-1),
            dim=1,
        )
        base_z = torch.sum(confidence * depth_tokens, dim=1)

        bbox = self._bbox(inputs, tokens.dtype).to(tokens.device)
        intrinsics = self._intrinsics(
            inputs,
            batch_size,
            tokens.device,
            tokens.dtype,
        )
        geometry, _, _, grid_side = self._lift_tokens(
            tokens,
            depth_tokens,
            bbox,
            intrinsics,
        )
        context = self.context_norm(tokens + self.geometry_embed(geometry))

        return {
            "rgb": rgb,
            "batch_size": batch_size,
            "tokens": tokens,
            "global_feature": global_feature,
            "depth_tokens": depth_tokens,
            "depth_confidence": confidence,
            "base_z": base_z,
            "bbox": bbox,
            "intrinsics": intrinsics,
            "geometry": geometry,
            "context": context,
            "grid_side": grid_side,
        }

    def decode_pose_features(
        self,
        encoded: dict[str, Any],
        category: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode one category-conditioned pose from shared encoded features."""
        tokens = encoded["tokens"]
        global_feature = encoded["global_feature"]
        context = encoded["context"]
        bbox = encoded["bbox"]
        intrinsics = encoded["intrinsics"]
        base_z = encoded["base_z"]
        depth_tokens = encoded["depth_tokens"]
        confidence = encoded["depth_confidence"]
        grid_side = int(encoded["grid_side"])
        rgb = encoded["rgb"]
        batch_size = int(encoded["batch_size"])

        category = category.to(device=tokens.device).reshape(-1).long()
        if category.shape[0] != batch_size:
            raise ValueError("category hypothesis count must match encoded batch size")
        if torch.any((category < 0) | (category >= self.num_classes)):
            raise ValueError("category_label lies outside the configured class range")
        query = self.query_projection(
            torch.cat([global_feature, self.category_embedding(category)], dim=-1)
        ).unsqueeze(1)
        for layer in self.decoder_layers:
            query = layer(query, context)
        query = query.squeeze(1)
        translation_feature = query
        size_feature = query
        if self.decouple_translation_size:
            branch_input = query.detach() if self.detach_pose_query_for_ts else query
            translation_feature = self.translation_feature_adapter(branch_input)
            size_feature = self.size_feature_adapter(branch_input)

        y1, x1, y2, x2 = bbox.unbind(dim=1)
        bbox_width = (x2 - x1).clamp_min(1.0)
        bbox_height = (y2 - y1).clamp_min(1.0)
        bbox_center = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dim=1)
        offset_ratio = self.max_uv_offset_ratio * torch.tanh(
            self.center_offset_head(translation_feature)
        )
        center_uv = bbox_center + offset_ratio * torch.stack(
            [bbox_width, bbox_height], dim=1
        )

        log_z_residual = self.max_log_z_residual * torch.tanh(
            self.log_z_residual_head(translation_feature).squeeze(-1)
        )
        z = (base_z * torch.exp(log_z_residual)).clamp(
            min=self.depth_min,
            max=self.depth_max,
        )
        fx, fy, cx, cy = intrinsics
        translation = torch.stack(
            [
                (center_uv[:, 0] - cx) * z / fx,
                (center_uv[:, 1] - cy) * z / fy,
                z,
            ],
            dim=1,
        )

        rotation_6d = self.rotation_head(query)
        rotation = six_d_to_rotation_matrix(rotation_6d)
        size = F.softplus(self.size_head(size_feature)) + self.size_min
        depth_map = F.interpolate(
            depth_tokens.reshape(batch_size, 1, grid_side, grid_side),
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return {
            "pred_translation": translation,
            "pred_rotation": rotation,
            "pred_size": size,
            "pred_direct_rotation_6d": rotation_6d,
            "pred_metric_size": size,
            "pred_metric_depth_z": z,
            "pred_e2e_surface_depth_z": base_z,
            "pred_e2e_depth": depth_map,
            "pred_e2e_depth_tokens": depth_tokens,
            "pred_e2e_depth_confidence": confidence,
            "pred_e2e_center_uv": center_uv,
            "pred_e2e_log_z_residual": log_z_residual,
            "pred_pose_query": query,
            "pred_pose_category": category,
            "pose_prediction_mode": "e2e_direct_pose",
            "translation_prediction_mode": "e2e_projective_uvz",
        }

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        start = time.time()
        encoded = self.encode_pose_features(inputs)
        category = self._category_indices(inputs, int(encoded["batch_size"]))
        outputs = self.decode_pose_features(encoded, category)
        outputs.update(
            {
                "resnet_forward_time": time.time() - start,
                "octree_forward_time": 0.0,
                "shapenet_forward_time": 0.0,
                "diffusion_forward_time": 0.0,
                "denoiser_time": 0.0,
            }
        )
        return outputs
