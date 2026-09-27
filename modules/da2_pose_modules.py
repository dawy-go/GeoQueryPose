from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.amp_utils import autocast_disabled
from utils.rotation_utils import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_6d,
    six_d_to_rotation_matrix,
)


class DA2VisualEncoder(nn.Module):
    """Depth Anything V2 backbone with trainable multi-scale token adapters.

    The NOCS input pipeline already applies ImageNet normalization, so this
    module only resizes the crop to a patch-aligned square before calling the
    Hugging Face DA2 backbone.
    """

    def __init__(
        self,
        model_name: str,
        input_size: int = 224,
        patch_size: int = 14,
        token_dim: int = 256,
        token_grid: int = 8,
        num_scales: int = 4,
        num_heads: int = 8,
        freeze_backbone: bool = True,
        local_files_only: bool = False,
        backbone: Optional[nn.Module] = None,
        backbone_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.token_dim = int(token_dim)
        self.token_grid = int(token_grid)
        self.num_scales = int(num_scales)
        self.freeze_backbone = bool(freeze_backbone)

        if self.input_size <= 0 or self.input_size % self.patch_size != 0:
            raise ValueError(
                f"DA2 input_size must be positive and divisible by patch_size; "
                f"got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        if self.token_grid <= 0:
            raise ValueError("DA2 token_grid must be positive")
        if self.token_dim % int(num_heads) != 0:
            raise ValueError("DA2 token_dim must be divisible by num_heads")

        if backbone is None:
            backbone, inferred_hidden_size, inferred_patch_size = self._load_backbone(
                self.model_name,
                local_files_only=bool(local_files_only),
            )
            backbone_hidden_size = inferred_hidden_size
            if inferred_patch_size is not None:
                self.patch_size = int(inferred_patch_size)
        elif backbone_hidden_size is None:
            raise ValueError("backbone_hidden_size is required with an injected backbone")

        self.backbone = backbone
        self.backbone_hidden_size = int(backbone_hidden_size)
        self.scale_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.backbone_hidden_size),
                    nn.Linear(self.backbone_hidden_size, self.token_dim),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.scale_embeddings = nn.Parameter(torch.zeros(self.num_scales, 1, 1, self.token_dim))
        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))
        self.spatial_position = nn.Parameter(
            torch.zeros(1, self.token_grid * self.token_grid, self.token_dim)
        )
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.global_query = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.global_attention = nn.MultiheadAttention(
            self.token_dim,
            num_heads=int(num_heads),
            batch_first=True,
        )
        self.global_norm = nn.LayerNorm(self.token_dim)

        nn.init.normal_(self.scale_embeddings, std=0.02)
        nn.init.normal_(self.spatial_position, std=0.02)
        nn.init.normal_(self.global_query, std=0.02)

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @staticmethod
    def _load_backbone(model_name: str, local_files_only: bool):
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as exc:
            raise ImportError(
                "DA2VisualEncoder requires transformers. Install the repository "
                "requirements before constructing model_arch=da2_octree."
            ) from exc

        depth_model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        config = depth_model.config
        backbone_config = getattr(config, "backbone_config", None)
        hidden_size = getattr(backbone_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(backbone_config, "embed_dim", None)
        if hidden_size is None:
            raise ValueError(f"Unable to determine DA2 backbone hidden size for {model_name}")
        patch_size = getattr(config, "patch_size", None)
        backbone = depth_model.backbone
        del depth_model
        return backbone, int(hidden_size), patch_size

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _backbone_feature_maps(self, pixel_values: torch.Tensor):
        grad_context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with grad_context:
            if hasattr(self.backbone, "forward_with_filtered_kwargs"):
                outputs = self.backbone.forward_with_filtered_kwargs(
                    pixel_values,
                    output_hidden_states=False,
                    output_attentions=False,
                )
            else:
                outputs = self.backbone(pixel_values)
        feature_maps = getattr(outputs, "feature_maps", None)
        if feature_maps is None and isinstance(outputs, (tuple, list)):
            feature_maps = outputs
        if feature_maps is None:
            raise ValueError("DA2 backbone did not return feature_maps")
        return list(feature_maps)

    @staticmethod
    def _infer_token_grid(token_count: int) -> tuple[int, int]:
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError(f"Cannot infer a square grid from {token_count} DA2 tokens")
        return side, side

    def _as_grid(self, feature: torch.Tensor, expected_hw: tuple[int, int]) -> torch.Tensor:
        if feature.ndim == 4:
            return feature
        if feature.ndim != 3:
            raise ValueError(f"Expected DA2 feature map with rank 3 or 4, got {feature.shape}")

        batch_size, token_count, channels = feature.shape
        expected_count = expected_hw[0] * expected_hw[1]
        if token_count > expected_count:
            # DINO-family backbones may retain a class or register token. Patch
            # tokens are kept at the end of the sequence.
            feature = feature[:, -expected_count:, :]
            token_count = expected_count
        if token_count == expected_count:
            height, width = expected_hw
        else:
            height, width = self._infer_token_grid(token_count)
        return feature.transpose(1, 2).reshape(batch_size, channels, height, width)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected ImageNet-normalized RGB [B,3,H,W], got {rgb.shape}")

        pixel_values = F.interpolate(
            rgb,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        feature_maps = self._backbone_feature_maps(pixel_values)
        if len(feature_maps) < self.num_scales:
            raise ValueError(
                f"DA2 backbone returned {len(feature_maps)} feature maps, "
                f"but num_scales={self.num_scales}"
            )
        feature_maps = feature_maps[-self.num_scales :]

        patch_hw = (
            self.input_size // self.patch_size,
            self.input_size // self.patch_size,
        )
        projected_grids = []
        for index, (feature, projection) in enumerate(
            zip(feature_maps, self.scale_projections)
        ):
            grid = self._as_grid(feature, patch_hw)
            tokens = grid.flatten(2).transpose(1, 2)
            tokens = projection(tokens) + self.scale_embeddings[index]
            grid = tokens.transpose(1, 2).reshape(
                tokens.shape[0],
                self.token_dim,
                grid.shape[-2],
                grid.shape[-1],
            )
            projected_grids.append(
                F.adaptive_avg_pool2d(grid, (self.token_grid, self.token_grid))
            )

        weights = torch.softmax(self.scale_logits, dim=0)
        fused_grid = sum(
            weight * grid for weight, grid in zip(weights, projected_grids)
        )
        spatial_tokens = fused_grid.flatten(2).transpose(1, 2)
        spatial_tokens = self.token_norm(spatial_tokens + self.spatial_position)

        query = self.global_query.expand(rgb.shape[0], -1, -1)
        global_token, _ = self.global_attention(
            query=query,
            key=spatial_tokens,
            value=spatial_tokens,
            need_weights=False,
        )
        global_token = self.global_norm(global_token.squeeze(1))
        return {
            "tokens": spatial_tokens,
            "global": global_token,
            "scale_weights": weights,
        }


class GeometryVisualFusion(nn.Module):
    """Use the Octree feature as a query over DA2 spatial tokens."""

    def __init__(self, octree_dim: int, token_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.octree_projection = nn.Sequential(
            nn.LayerNorm(int(octree_dim)),
            nn.Linear(int(octree_dim), int(token_dim)),
        )
        self.query_norm = nn.LayerNorm(int(token_dim))
        self.visual_norm = nn.LayerNorm(int(token_dim))
        self.attention = nn.MultiheadAttention(
            int(token_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(int(token_dim))
        self.ffn = nn.Sequential(
            nn.Linear(int(token_dim), int(token_dim) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(token_dim) * 4, int(token_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        octree_feature: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        octree_token = self.octree_projection(octree_feature)
        query = self.query_norm(octree_token).unsqueeze(1)
        attended, _ = self.attention(
            query=query,
            key=self.visual_norm(visual_tokens),
            value=self.visual_norm(visual_tokens),
            need_weights=False,
        )
        fused = octree_token + attended.squeeze(1)
        fused = self.output_norm(fused + self.ffn(self.output_norm(fused)))
        return octree_token, fused


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim * 4),
            nn.SiLU(),
            nn.Linear(self.output_dim * 4, self.output_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.output_dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.output_dim:
            embedding = F.pad(embedding, (0, self.output_dim - embedding.shape[1]))
        return self.projection(embedding)


class PoseContextEncoder(nn.Module):
    """Build a real token sequence instead of a single concatenated token."""

    def __init__(
        self,
        token_dim: int,
        num_classes: int = 6,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        token_dim = int(token_dim)
        self.time_embedding = SinusoidalTimeEmbedding(token_dim)
        self.category_embedding = nn.Embedding(int(num_classes), token_dim)
        # time, DA global, Octree, fused shape, category, DA spatial
        self.modality_embedding = nn.Parameter(torch.zeros(6, 1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=int(num_heads),
            dim_feedforward=token_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(token_dim),
        )
        nn.init.normal_(self.modality_embedding, std=0.02)

    def category_token(self, category: torch.Tensor) -> torch.Tensor:
        return self.category_embedding(category)

    def forward(
        self,
        timesteps: torch.Tensor,
        da_global: torch.Tensor,
        octree_token: torch.Tensor,
        fused_token: torch.Tensor,
        category: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tokens = [
            self.time_embedding(timesteps).unsqueeze(1) + self.modality_embedding[0],
            da_global.unsqueeze(1) + self.modality_embedding[1],
            octree_token.unsqueeze(1) + self.modality_embedding[2],
            fused_token.unsqueeze(1) + self.modality_embedding[3],
            self.category_token(category).unsqueeze(1) + self.modality_embedding[4],
            spatial_tokens + self.modality_embedding[5],
        ]
        return self.encoder(torch.cat(tokens, dim=1))


class StaticPoseContextEncoder(nn.Module):
    """Encode DA2/Octree/category tokens without a diffusion timestep.

    This encoder belongs exclusively to the structured deterministic R/S route.
    It intentionally has its own category and modality embeddings so a selective
    warm start can preserve the shared feature extractors while learning the
    complete R/S mapping from fresh parameters.
    """

    def __init__(
        self,
        token_dim: int,
        num_classes: int = 6,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        token_dim = int(token_dim)
        self.category_embedding = nn.Embedding(int(num_classes), token_dim)
        # DA global, Octree, fused shape, category, DA spatial
        self.modality_embedding = nn.Parameter(torch.zeros(5, 1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=int(num_heads),
            dim_feedforward=token_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(token_dim),
        )
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(
        self,
        da_global: torch.Tensor,
        octree_token: torch.Tensor,
        fused_token: torch.Tensor,
        category: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tokens = [
            da_global.unsqueeze(1) + self.modality_embedding[0],
            octree_token.unsqueeze(1) + self.modality_embedding[1],
            fused_token.unsqueeze(1) + self.modality_embedding[2],
            self.category_embedding(category).unsqueeze(1)
            + self.modality_embedding[3],
            spatial_tokens + self.modality_embedding[4],
        ]
        return self.encoder(torch.cat(tokens, dim=1))


class PoseCrossAttentionBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(int(token_dim))
        self.context_norm = nn.LayerNorm(int(token_dim))
        self.cross_attention = nn.MultiheadAttention(
            int(token_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(int(token_dim))
        self.ffn = nn.Sequential(
            nn.Linear(int(token_dim), int(token_dim) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(token_dim) * 4, int(token_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=self.query_norm(query),
            key=self.context_norm(context),
            value=self.context_norm(context),
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class DeterministicPoseQuery(nn.Module):
    """A learned task query that reads the full static pose context once."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        token_dim = int(token_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.blocks = nn.ModuleList(
            [
                PoseCrossAttentionBlock(token_dim, num_heads, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(token_dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, context)
        return self.output_norm(query.squeeze(1))


class PoseDiffusionBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pose_embedding = nn.Sequential(
            nn.Linear(int(input_dim), int(token_dim)),
            nn.LayerNorm(int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.branch_token = nn.Parameter(torch.zeros(1, 1, int(token_dim)))
        self.blocks = nn.ModuleList(
            [
                PoseCrossAttentionBlock(token_dim, num_heads, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim) * 2),
            nn.SiLU(),
            nn.Linear(int(token_dim) * 2, int(output_dim)),
        )
        nn.init.normal_(self.branch_token, std=0.02)

    def forward(self, noisy_pose: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = self.pose_embedding(noisy_pose).unsqueeze(1) + self.branch_token
        for block in self.blocks:
            query = block(query, context)
        return self.output(query.squeeze(1))


class DeterministicTranslationBranch(nn.Module):
    """Predict image-center and metric-depth corrections without diffusion.

    The branch uses an explicit geometry query and reads the same DA2/Octree
    token context as the diffusion branches.  Its final layer is initialized to
    zero, so a checkpoint trained before this branch existed initially falls
    back exactly to ``base_translation``.
    """

    def __init__(
        self,
        geometry_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_center_offset_ratio: float = 0.5,
        max_log_depth_residual: float = 0.7,
        min_depth: float = 0.1,
        max_depth: float = 3.0,
    ) -> None:
        super().__init__()
        self.geometry_dim = int(geometry_dim)
        self.max_center_offset_ratio = float(max_center_offset_ratio)
        self.max_log_depth_residual = float(max_log_depth_residual)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        if self.geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive")
        if self.max_center_offset_ratio < 0.0:
            raise ValueError("max_center_offset_ratio must be non-negative")
        if self.max_log_depth_residual < 0.0:
            raise ValueError("max_log_depth_residual must be non-negative")
        if self.min_depth <= 0.0 or self.max_depth <= self.min_depth:
            raise ValueError("deterministic translation depth range is invalid")

        self.geometry_embedding = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            nn.Linear(self.geometry_dim, int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.branch_token = nn.Parameter(torch.zeros(1, 1, int(token_dim)))
        self.blocks = nn.ModuleList(
            [
                PoseCrossAttentionBlock(token_dim, num_heads, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim) * 2),
            nn.SiLU(),
            nn.Linear(int(token_dim) * 2, 3),
        )
        nn.init.normal_(self.branch_token, std=0.02)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        geometry_features: torch.Tensor,
        context: torch.Tensor,
        base_translation: torch.Tensor,
        base_uv: torch.Tensor,
        bbox_wh: torch.Tensor,
        cam_k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if geometry_features.shape[-1] != self.geometry_dim:
            raise ValueError(
                "deterministic translation geometry feature mismatch: "
                f"expected {self.geometry_dim}, got {geometry_features.shape[-1]}"
            )
        query = self.geometry_embedding(geometry_features).unsqueeze(1)
        query = query + self.branch_token
        for block in self.blocks:
            query = block(query, context)
        raw = self.output(query.squeeze(1))

        center_offset_normalized = (
            self.max_center_offset_ratio * torch.tanh(raw[:, :2])
        )
        center_uv = base_uv + center_offset_normalized * bbox_wh.clamp_min(1.0)
        log_depth_residual = (
            self.max_log_depth_residual * torch.tanh(raw[:, 2])
        )
        base_z = base_translation[:, 2].clamp_min(self.min_depth)
        depth = (base_z * torch.exp(log_depth_residual)).clamp(
            min=self.min_depth,
            max=self.max_depth,
        )

        fx, fy, cx, cy = cam_k.unbind(dim=-1)
        x = (center_uv[:, 0] - cx) * depth / fx.clamp_min(1.0e-6)
        y = (center_uv[:, 1] - cy) * depth / fy.clamp_min(1.0e-6)
        translation = torch.stack([x, y, depth], dim=-1)
        return {
            "translation": translation,
            "center_uv": center_uv,
            "center_offset_normalized": center_offset_normalized,
            "log_depth_residual": log_depth_residual,
            "depth": depth,
            "raw": raw,
        }


class DepthSetEncoder(nn.Module):
    """Encode the valid metric point set without collapsing it to moments."""

    def __init__(
        self,
        point_dim: int,
        token_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing_tokens: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        point_dim = int(point_dim)
        token_dim = int(token_dim)
        if point_dim <= 0 or token_dim <= 0:
            raise ValueError("DepthSetEncoder dimensions must be positive")
        if token_dim % int(num_heads) != 0:
            raise ValueError("DepthSetEncoder token_dim must divide num_heads")
        if int(num_inducing_tokens) <= 0:
            raise ValueError("DepthSetEncoder needs positive inducing tokens")

        self.point_dim = point_dim
        self.input_projection = nn.Sequential(
            nn.LayerNorm(point_dim),
            nn.Linear(point_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.depth_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.inducing_tokens = nn.Parameter(
            torch.zeros(1, int(num_inducing_tokens), token_dim)
        )
        self.inducing_query_norm = nn.LayerNorm(token_dim)
        self.point_norm = nn.LayerNorm(token_dim)
        self.inducing_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.inducing_ffn_norm = nn.LayerNorm(token_dim)
        self.inducing_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(token_dim * 4, token_dim),
            nn.Dropout(float(dropout)),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=int(num_heads),
            dim_feedforward=token_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(token_dim),
        )
        nn.init.normal_(self.depth_token, std=0.02)
        nn.init.normal_(self.inducing_tokens, std=0.02)

    def forward(
        self,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor,
        return_point_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if point_features.ndim != 3:
            raise ValueError("DepthSetEncoder expects [B, N, C] point features")
        if point_features.shape[-1] != self.point_dim:
            raise ValueError(
                "DepthSetEncoder point feature mismatch: "
                f"expected {self.point_dim}, got {point_features.shape[-1]}"
            )
        valid_mask = valid_mask.to(device=point_features.device).bool()
        if valid_mask.shape != point_features.shape[:2]:
            raise ValueError("DepthSetEncoder valid mask must have shape [B, N]")

        with autocast_disabled(point_features.device):
            points = self.input_projection(point_features.float())
            # Avoid O(N^2) attention over 1024 points. A small learned set of
            # inducing queries reads every valid point in O(MN), then the
            # compact induced sequence is refined by the Transformer encoder.
            attention_valid = valid_mask.clone()
            no_valid = ~attention_valid.any(dim=1)
            if torch.any(no_valid):
                attention_valid[no_valid, 0] = True
                points = points.clone()
                points[no_valid, 0] = 0.0
            inducing = self.inducing_tokens.expand(points.shape[0], -1, -1)
            attended, _ = self.inducing_attention(
                query=self.inducing_query_norm(inducing),
                key=self.point_norm(points),
                value=self.point_norm(points),
                key_padding_mask=~attention_valid,
                need_weights=False,
            )
            inducing = inducing + attended
            inducing = inducing + self.inducing_ffn(
                self.inducing_ffn_norm(inducing)
            )
            depth_token = self.depth_token.expand(points.shape[0], -1, -1)
            tokens = torch.cat([depth_token, inducing], dim=1)
            encoded = self.encoder(tokens)
            global_token = encoded[:, 0]
            if return_point_tokens:
                return global_token, points
            return global_token


class RayDepthTranslationBranch(nn.Module):
    """Predict a bounded image ray and metric log-depth deterministically."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_center_offset_ratio: float = 0.5,
        max_log_depth_residual: float = 0.35,
        min_depth: float = 0.1,
        max_depth: float = 3.0,
    ) -> None:
        super().__init__()
        self.max_center_offset_ratio = float(max_center_offset_ratio)
        self.max_log_depth_residual = float(max_log_depth_residual)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        if self.max_center_offset_ratio < 0.0:
            raise ValueError("ray-depth center offset bound must be non-negative")
        if self.max_log_depth_residual < 0.0:
            raise ValueError("ray-depth log-depth bound must be non-negative")
        if self.min_depth <= 0.0 or self.max_depth <= self.min_depth:
            raise ValueError("ray-depth metric range is invalid")

        self.depth_context_projection = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.center_query = DeterministicPoseQuery(
            token_dim, num_heads, num_layers, dropout
        )
        self.depth_query = DeterministicPoseQuery(
            token_dim, num_heads, num_layers, dropout
        )
        self.center_output = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), 2),
        )
        self.depth_output = nn.Sequential(
            nn.LayerNorm(int(token_dim) * 2),
            nn.Linear(int(token_dim) * 2, int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), 1),
        )
        # Tiny weights preserve the robust depth/ray anchor while allowing the
        # point encoder and both queries to receive gradients on step one.
        nn.init.normal_(self.center_output[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.center_output[-1].bias)
        nn.init.normal_(self.depth_output[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.depth_output[-1].bias)

    @staticmethod
    def _backproject(
        center_uv: torch.Tensor,
        depth: torch.Tensor,
        cam_k: torch.Tensor,
    ) -> torch.Tensor:
        fx, fy, cx, cy = cam_k.unbind(dim=-1)
        x = (center_uv[:, 0] - cx) * depth / fx.clamp_min(1.0e-6)
        y = (center_uv[:, 1] - cy) * depth / fy.clamp_min(1.0e-6)
        return torch.stack([x, y, depth], dim=-1)

    def forward(
        self,
        depth_token: torch.Tensor,
        context: torch.Tensor,
        base_depth: torch.Tensor,
        base_uv: torch.Tensor,
        bbox_wh: torch.Tensor,
        cam_k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with autocast_disabled(depth_token.device):
            depth_token = self.depth_context_projection(depth_token.float())
            context = context.float()
            augmented_context = torch.cat(
                [depth_token.unsqueeze(1), context],
                dim=1,
            )
            center_feature = self.center_query(augmented_context)
            depth_feature = self.depth_query(augmented_context)
            center_raw = self.center_output(center_feature)
            depth_raw = self.depth_output(
                torch.cat([depth_feature, depth_token], dim=-1)
            ).squeeze(-1)

            center_offset_normalized = (
                self.max_center_offset_ratio * torch.tanh(center_raw)
            )
            center_uv = base_uv.float() + center_offset_normalized * bbox_wh.float().clamp_min(1.0)
            log_depth_residual = (
                self.max_log_depth_residual * torch.tanh(depth_raw)
            )
            base_depth = base_depth.float().clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            depth = (base_depth * torch.exp(log_depth_residual)).clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            translation = self._backproject(center_uv, depth, cam_k.float())
            anchor_translation = self._backproject(
                base_uv.float(),
                base_depth,
                cam_k.float(),
            )
            return {
                "translation": translation,
                "anchor_translation": anchor_translation,
                "center_uv": center_uv,
                "center_offset_normalized": center_offset_normalized,
                "log_depth_residual": log_depth_residual,
                "depth": depth,
                "base_depth": base_depth,
                "raw": torch.cat([center_raw, depth_raw.unsqueeze(-1)], dim=-1),
            }


class PointVoteTranslationBranch(nn.Module):
    """Deterministic point-to-center voting for image ray and metric depth.

    Every valid metric-depth point predicts a normalized image-center offset
    and a log-depth correction.  The final center/depth uses a fixed trimmed
    mean, so robustness does not depend on a learned gate.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_sources: int = 2,
        trim_ratio: float = 0.20,
        inference_max_center_offset_ratio: float = 1.0,
        inference_max_log_depth_residual: float = 0.70,
        min_depth: float = 0.1,
        max_depth: float = 3.0,
    ) -> None:
        super().__init__()
        self.trim_ratio = float(trim_ratio)
        self.inference_max_center_offset_ratio = float(
            inference_max_center_offset_ratio
        )
        self.inference_max_log_depth_residual = float(
            inference_max_log_depth_residual
        )
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        if not 0.0 <= self.trim_ratio < 0.5:
            raise ValueError("point-vote trim_ratio must be in [0, 0.5)")
        if int(num_sources) <= 0:
            raise ValueError("point-vote num_sources must be positive")
        if self.min_depth <= 0.0 or self.max_depth <= self.min_depth:
            raise ValueError("point-vote metric range is invalid")

        self.depth_context_projection = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.global_query = DeterministicPoseQuery(
            token_dim,
            num_heads,
            num_layers,
            dropout,
        )
        self.source_embedding = nn.Embedding(int(num_sources), int(token_dim))
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(int(token_dim) * 2),
            nn.Linear(int(token_dim) * 2, int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.vote_output = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), 3),
        )
        nn.init.normal_(self.vote_output[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.vote_output[-1].bias)

    @staticmethod
    def _backproject(
        center_uv: torch.Tensor,
        depth: torch.Tensor,
        cam_k: torch.Tensor,
    ) -> torch.Tensor:
        fx, fy, cx, cy = cam_k.unbind(dim=-1)
        x = (center_uv[:, 0] - cx) * depth / fx.clamp_min(1.0e-6)
        y = (center_uv[:, 1] - cy) * depth / fy.clamp_min(1.0e-6)
        return torch.stack([x, y, depth], dim=-1)

    @staticmethod
    def _project_points(
        points: torch.Tensor,
        cam_k: torch.Tensor,
    ) -> torch.Tensor:
        fx, fy, cx, cy = cam_k.unbind(dim=-1)
        depth = points[:, :, 2].clamp_min(1.0e-6)
        u = fx.unsqueeze(1) * points[:, :, 0] / depth + cx.unsqueeze(1)
        v = fy.unsqueeze(1) * points[:, :, 1] / depth + cy.unsqueeze(1)
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _masked_trimmed_mean(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        trim_ratio: float,
    ) -> torch.Tensor:
        """Coordinate-wise masked trimmed mean over point dimension."""
        if values.ndim not in {2, 3}:
            raise ValueError("trimmed mean expects [B,N] or [B,N,C]")
        if valid_mask.shape != values.shape[:2]:
            raise ValueError("trimmed mean mask must have shape [B,N]")
        squeeze = values.ndim == 2
        work = values.unsqueeze(-1) if squeeze else values
        valid = valid_mask.unsqueeze(-1) & torch.isfinite(work)
        ordered = work.masked_fill(~valid, float("inf")).sort(dim=1).values
        counts = valid.sum(dim=1)
        requested_trim = torch.floor(counts.float() * float(trim_ratio)).long()
        max_trim = ((counts - 1).clamp_min(0) // 2)
        trim = torch.minimum(requested_trim, max_trim)
        ranks = torch.arange(
            work.shape[1],
            device=work.device,
        ).view(1, -1, 1)
        keep = (ranks >= trim.unsqueeze(1)) & (
            ranks < (counts - trim).unsqueeze(1)
        )
        keep = keep & torch.isfinite(ordered)
        total = torch.where(keep, ordered, torch.zeros_like(ordered)).sum(dim=1)
        mean = total / keep.sum(dim=1).clamp_min(1).to(dtype=work.dtype)
        fallback = torch.nan_to_num(work, nan=0.0).mean(dim=1)
        mean = torch.where(counts > 0, mean, fallback)
        return mean.squeeze(-1) if squeeze else mean

    def forward(
        self,
        depth_token: torch.Tensor,
        point_tokens: torch.Tensor,
        context: torch.Tensor,
        source_indices: torch.Tensor,
        metric_points: torch.Tensor,
        valid_mask: torch.Tensor,
        base_depth: torch.Tensor,
        base_uv: torch.Tensor,
        bbox_wh: torch.Tensor,
        cam_k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with autocast_disabled(depth_token.device):
            depth_token = self.depth_context_projection(depth_token.float())
            point_tokens = point_tokens.float()
            context = context.float()
            metric_points = metric_points.float()
            valid_mask = valid_mask.to(device=metric_points.device).bool()
            source_indices = source_indices.to(device=metric_points.device).long()
            source_indices = source_indices.clamp(
                0,
                self.source_embedding.num_embeddings - 1,
            )

            augmented_context = torch.cat(
                [depth_token.unsqueeze(1), context],
                dim=1,
            )
            global_feature = self.global_query(augmented_context)
            source_feature = self.source_embedding(source_indices)
            condition = self.condition_projection(
                torch.cat([global_feature, source_feature], dim=-1)
            )
            vote_raw = self.vote_output(
                point_tokens + condition.unsqueeze(1)
            )

            point_uv = self._project_points(metric_points, cam_k.float())
            point_depth = metric_points[:, :, 2].clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            vote_center_offset = vote_raw[:, :, :2]
            vote_log_depth_residual = vote_raw[:, :, 2]
            if not self.training:
                if self.inference_max_center_offset_ratio > 0.0:
                    vote_center_offset = vote_center_offset.clamp(
                        -self.inference_max_center_offset_ratio,
                        self.inference_max_center_offset_ratio,
                    )
                if self.inference_max_log_depth_residual > 0.0:
                    vote_log_depth_residual = vote_log_depth_residual.clamp(
                        -self.inference_max_log_depth_residual,
                        self.inference_max_log_depth_residual,
                    )

            vote_uv = (
                point_uv
                + vote_center_offset * bbox_wh.float().clamp_min(1.0).unsqueeze(1)
            )
            vote_log_depth = (
                torch.log(point_depth.clamp_min(1.0e-6))
                + vote_log_depth_residual
            )
            center_uv = self._masked_trimmed_mean(
                vote_uv,
                valid_mask,
                self.trim_ratio,
            )
            log_depth = self._masked_trimmed_mean(
                vote_log_depth,
                valid_mask,
                self.trim_ratio,
            )
            depth = torch.exp(log_depth.clamp(-10.0, 10.0))
            if not self.training:
                depth = depth.clamp(
                    min=self.min_depth,
                    max=self.max_depth,
                )
            translation = self._backproject(center_uv, depth, cam_k.float())

            base_depth = base_depth.float().clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            base_uv = base_uv.float()
            bbox_wh = bbox_wh.float().clamp_min(1.0)
            center_offset_normalized = (center_uv - base_uv) / bbox_wh
            log_depth_residual = torch.log(
                depth / base_depth.clamp_min(1.0e-6)
            )
            anchor_translation = self._backproject(
                base_uv,
                base_depth,
                cam_k.float(),
            )
            return {
                "translation": translation,
                "anchor_translation": anchor_translation,
                "center_uv": center_uv,
                "center_offset_normalized": center_offset_normalized,
                "log_depth_residual": log_depth_residual,
                "depth": depth,
                "base_depth": base_depth,
                "vote_uv": vote_uv,
                "vote_log_depth": vote_log_depth,
                "vote_valid_mask": valid_mask,
                "point_uv": point_uv,
                "point_depth": point_depth,
                "vote_center_offset_normalized": vote_center_offset,
                "vote_log_depth_residual": vote_log_depth_residual,
                "raw": self._masked_trimmed_mean(
                    vote_raw,
                    valid_mask,
                    self.trim_ratio,
                ),
            }


class PointVoteTranslationV3Branch(PointVoteTranslationBranch):
    """Geometry-conditioned uncertainty voting with a coarse-depth fallback.

    The inherited V2 modules intentionally keep the same names so an epoch-48
    V2 checkpoint can initialize this branch tensor-for-tensor.  Every V3
    addition is zero initialized: geometry starts as a no-op, uncertainty
    starts with uniform vote weights, and the fallback starts at zero.  This
    makes an untrained V3 branch reproduce V2 before the isolated additions are
    optimized.
    """

    _V3_MODULE_NAMES = (
        "v3_geometry_pose_projection",
        "v3_uncertainty_output",
        "v3_gate_stats_projection",
        "v3_fallback_gate",
    )

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_sources: int = 2,
        trim_ratio: float = 0.20,
        inference_max_center_offset_ratio: float = 1.0,
        inference_max_log_depth_residual: float = 0.70,
        min_depth: float = 0.1,
        max_depth: float = 3.0,
        geometry_dim: int = 18,
        structured_pose_dim: int = 9,
        max_log_variance: float = 4.0,
        max_fallback_weight: float = 0.75,
    ) -> None:
        super().__init__(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_sources=num_sources,
            trim_ratio=trim_ratio,
            inference_max_center_offset_ratio=(
                inference_max_center_offset_ratio
            ),
            inference_max_log_depth_residual=(
                inference_max_log_depth_residual
            ),
            min_depth=min_depth,
            max_depth=max_depth,
        )
        self.geometry_dim = int(geometry_dim)
        self.structured_pose_dim = int(structured_pose_dim)
        self.max_log_variance = float(max_log_variance)
        self.max_fallback_weight = float(max_fallback_weight)
        if self.geometry_dim <= 0:
            raise ValueError("V3 geometry_dim must be positive")
        if self.structured_pose_dim <= 0:
            raise ValueError("V3 structured_pose_dim must be positive")
        if self.max_log_variance <= 0.0:
            raise ValueError("V3 max_log_variance must be positive")
        if not 0.0 <= self.max_fallback_weight <= 1.0:
            raise ValueError("V3 max_fallback_weight must be in [0, 1]")

        pose_geometry_dim = self.geometry_dim + self.structured_pose_dim
        self.v3_geometry_pose_projection = nn.Sequential(
            nn.LayerNorm(pose_geometry_dim),
            nn.Linear(pose_geometry_dim, int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.v3_uncertainty_output = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), 2),
        )
        # uv RMS (2), log-z RMS (1), mean log variances (2), valid ratio (1).
        self.v3_gate_stats_projection = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.v3_fallback_gate = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), 1),
        )

        # Preserve V2 exactly at warm-start.  Earlier layers still receive
        # gradients after their zero-initialized output layers move away from 0.
        for module in (
            self.v3_geometry_pose_projection,
            self.v3_uncertainty_output,
            self.v3_gate_stats_projection,
            self.v3_fallback_gate,
        ):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        self._train_additions_only = False

    def _v3_modules(self) -> tuple[nn.Module, ...]:
        return tuple(getattr(self, name) for name in self._V3_MODULE_NAMES)

    def freeze_v2_base(self) -> None:
        """Freeze inherited V2 tensors and train only zero-init V3 additions."""
        self.requires_grad_(False)
        for module in self._v3_modules():
            module.requires_grad_(True)
        self._train_additions_only = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._train_additions_only:
            # Frozen V2 dropout must remain deterministic during the controlled
            # warm-start while the four V3 additions train normally.
            for name, module in self.named_children():
                if name not in self._V3_MODULE_NAMES:
                    module.eval()
            for module in self._v3_modules():
                module.train(True)
        return self

    @staticmethod
    def _masked_weighted_trimmed_mean(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        weights: torch.Tensor,
        trim_ratio: float,
    ) -> torch.Tensor:
        """Coordinate-wise trimmed mean with learned positive vote weights."""
        if values.ndim not in {2, 3}:
            raise ValueError("weighted trimmed mean expects [B,N] or [B,N,C]")
        if valid_mask.shape != values.shape[:2]:
            raise ValueError("weighted trimmed mean mask must have shape [B,N]")
        squeeze = values.ndim == 2
        work = values.unsqueeze(-1) if squeeze else values
        if weights.ndim == 2:
            vote_weights = weights.unsqueeze(-1)
        elif weights.ndim == 3:
            vote_weights = weights
        else:
            raise ValueError("weighted trimmed mean weights need rank 2 or 3")
        if vote_weights.shape[:2] != work.shape[:2]:
            raise ValueError("weighted trimmed mean weight shape mismatch")
        if vote_weights.shape[-1] == 1 and work.shape[-1] != 1:
            vote_weights = vote_weights.expand(-1, -1, work.shape[-1])
        if vote_weights.shape != work.shape:
            raise ValueError("weighted trimmed mean channel mismatch")

        valid = valid_mask.unsqueeze(-1) & torch.isfinite(work)
        sortable = work.masked_fill(~valid, float("inf"))
        ordered, indices = sortable.sort(dim=1)
        ordered_weights = vote_weights.gather(1, indices).clamp_min(0.0)
        ordered_weights = torch.where(
            torch.isfinite(ordered), ordered_weights, torch.zeros_like(ordered_weights)
        )
        counts = valid.sum(dim=1)
        requested_trim = torch.floor(counts.float() * float(trim_ratio)).long()
        max_trim = ((counts - 1).clamp_min(0) // 2)
        trim = torch.minimum(requested_trim, max_trim)
        ranks = torch.arange(work.shape[1], device=work.device).view(1, -1, 1)
        keep = (ranks >= trim.unsqueeze(1)) & (
            ranks < (counts - trim).unsqueeze(1)
        )
        keep = keep & torch.isfinite(ordered)
        kept_weights = torch.where(keep, ordered_weights, torch.zeros_like(ordered_weights))
        total = torch.where(keep, ordered, torch.zeros_like(ordered))
        total = (total * kept_weights).sum(dim=1)
        denominator = kept_weights.sum(dim=1)
        weighted_mean = total / denominator.clamp_min(1.0e-6)

        fallback_weights = torch.where(
            valid, vote_weights.clamp_min(0.0), torch.zeros_like(vote_weights)
        )
        fallback = (
            torch.where(valid, work, torch.zeros_like(work)) * fallback_weights
        ).sum(dim=1) / fallback_weights.sum(dim=1).clamp_min(1.0e-6)
        mean = torch.where(denominator > 0.0, weighted_mean, fallback)
        return mean.squeeze(-1) if squeeze else mean

    @staticmethod
    def _project_translation(
        translation: torch.Tensor,
        cam_k: torch.Tensor,
    ) -> torch.Tensor:
        fx, fy, cx, cy = cam_k.unbind(dim=-1)
        depth = translation[:, 2].clamp_min(1.0e-6)
        return torch.stack(
            [
                fx * translation[:, 0] / depth + cx,
                fy * translation[:, 1] / depth + cy,
            ],
            dim=-1,
        )

    def forward(
        self,
        depth_token: torch.Tensor,
        point_tokens: torch.Tensor,
        context: torch.Tensor,
        source_indices: torch.Tensor,
        metric_points: torch.Tensor,
        valid_mask: torch.Tensor,
        base_depth: torch.Tensor,
        base_uv: torch.Tensor,
        bbox_wh: torch.Tensor,
        cam_k: torch.Tensor,
        geometry_features: torch.Tensor,
        structured_size: torch.Tensor,
        structured_rotation_6d: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with autocast_disabled(depth_token.device):
            depth_token = self.depth_context_projection(depth_token.float())
            point_tokens = point_tokens.float()
            context = context.float()
            metric_points = metric_points.float()
            valid_mask = valid_mask.to(device=metric_points.device).bool()
            source_indices = source_indices.to(device=metric_points.device).long()
            source_indices = source_indices.clamp(
                0,
                self.source_embedding.num_embeddings - 1,
            )

            geometry_features = geometry_features.float()
            structured_pose = torch.cat(
                [
                    torch.log(structured_size.float().clamp_min(1.0e-6)),
                    structured_rotation_6d.float(),
                ],
                dim=-1,
            )
            if geometry_features.shape[-1] != self.geometry_dim:
                raise ValueError(
                    "V3 geometry feature mismatch: expected {}, got {}".format(
                        self.geometry_dim, geometry_features.shape[-1]
                    )
                )
            if structured_pose.shape[-1] != self.structured_pose_dim:
                raise ValueError(
                    "V3 structured pose mismatch: expected {}, got {}".format(
                        self.structured_pose_dim, structured_pose.shape[-1]
                    )
                )

            augmented_context = torch.cat(
                [depth_token.unsqueeze(1), context],
                dim=1,
            )
            global_feature = self.global_query(augmented_context)
            source_feature = self.source_embedding(source_indices)
            condition = self.condition_projection(
                torch.cat([global_feature, source_feature], dim=-1)
            )
            geometry_delta = self.v3_geometry_pose_projection(
                torch.cat([geometry_features, structured_pose], dim=-1)
            )
            condition = condition + geometry_delta
            vote_features = point_tokens + condition.unsqueeze(1)
            vote_raw = self.vote_output(vote_features)
            vote_log_variance = self.v3_uncertainty_output(vote_features).clamp(
                -self.max_log_variance,
                self.max_log_variance,
            )
            uv_weights = torch.exp(-vote_log_variance[:, :, :1])
            depth_weights = torch.exp(-vote_log_variance[:, :, 1])

            point_uv = self._project_points(metric_points, cam_k.float())
            point_depth = metric_points[:, :, 2].clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            vote_center_offset = vote_raw[:, :, :2]
            vote_log_depth_residual = vote_raw[:, :, 2]
            if not self.training:
                if self.inference_max_center_offset_ratio > 0.0:
                    vote_center_offset = vote_center_offset.clamp(
                        -self.inference_max_center_offset_ratio,
                        self.inference_max_center_offset_ratio,
                    )
                if self.inference_max_log_depth_residual > 0.0:
                    vote_log_depth_residual = vote_log_depth_residual.clamp(
                        -self.inference_max_log_depth_residual,
                        self.inference_max_log_depth_residual,
                    )

            bbox_wh = bbox_wh.float().clamp_min(1.0)
            vote_uv = point_uv + vote_center_offset * bbox_wh.unsqueeze(1)
            vote_log_depth = (
                torch.log(point_depth.clamp_min(1.0e-6))
                + vote_log_depth_residual
            )
            point_center_uv = self._masked_weighted_trimmed_mean(
                vote_uv,
                valid_mask,
                uv_weights,
                self.trim_ratio,
            )
            point_log_depth = self._masked_weighted_trimmed_mean(
                vote_log_depth,
                valid_mask,
                depth_weights,
                self.trim_ratio,
            )
            point_depth_prediction = torch.exp(
                point_log_depth.clamp(-10.0, 10.0)
            )
            if not self.training:
                point_depth_prediction = point_depth_prediction.clamp(
                    min=self.min_depth,
                    max=self.max_depth,
                )
            point_translation = self._backproject(
                point_center_uv,
                point_depth_prediction,
                cam_k.float(),
            )

            valid_f = valid_mask.to(dtype=point_translation.dtype)
            denominator = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            uv_delta = (vote_uv - point_center_uv.unsqueeze(1)) / bbox_wh.unsqueeze(1)
            log_depth_delta = vote_log_depth - point_log_depth.unsqueeze(1)
            uv_rms = torch.sqrt(
                ((uv_delta ** 2) * valid_f.unsqueeze(-1)).sum(dim=1)
                / denominator
                + 1.0e-8
            )
            log_depth_rms = torch.sqrt(
                ((log_depth_delta ** 2) * valid_f).sum(dim=1, keepdim=True)
                / denominator
                + 1.0e-8
            )
            mean_log_variance = (
                vote_log_variance * valid_f.unsqueeze(-1)
            ).sum(dim=1) / denominator
            valid_ratio = valid_f.mean(dim=1, keepdim=True)
            gate_stats = torch.cat(
                [uv_rms, log_depth_rms, mean_log_variance, valid_ratio],
                dim=-1,
            )
            gate_feature = condition + self.v3_gate_stats_projection(gate_stats)
            fallback_raw = self.v3_fallback_gate(gate_feature).squeeze(-1)
            # softplus(0)-log(2)=0 with a non-zero derivative, so the V3
            # warm-start is exactly V2 while the gate can learn on step one.
            fallback_weight = (
                F.softplus(fallback_raw) - math.log(2.0)
            ).clamp(min=0.0, max=self.max_fallback_weight)

            base_depth = base_depth.float().clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            base_uv = base_uv.float()
            anchor_translation = self._backproject(
                base_uv,
                base_depth,
                cam_k.float(),
            )
            translation = point_translation + fallback_weight.unsqueeze(-1) * (
                anchor_translation - point_translation
            )
            depth = translation[:, 2]
            center_uv = self._project_translation(translation, cam_k.float())
            center_offset_normalized = (center_uv - base_uv) / bbox_wh
            log_depth_residual = torch.log(
                depth.clamp_min(1.0e-6) / base_depth.clamp_min(1.0e-6)
            )
            return {
                "translation": translation,
                "anchor_translation": anchor_translation,
                "center_uv": center_uv,
                "center_offset_normalized": center_offset_normalized,
                "log_depth_residual": log_depth_residual,
                "depth": depth,
                "base_depth": base_depth,
                "vote_uv": vote_uv,
                "vote_log_depth": vote_log_depth,
                "vote_valid_mask": valid_mask,
                "point_uv": point_uv,
                "point_depth": point_depth,
                "vote_center_offset_normalized": vote_center_offset,
                "vote_log_depth_residual": vote_log_depth_residual,
                "vote_log_variance": vote_log_variance,
                "vote_confidence": torch.sigmoid(-vote_log_variance),
                "vote_dispersion": gate_stats,
                "fallback_weight": fallback_weight,
                "point_translation": point_translation,
                "point_center_uv": point_center_uv,
                "point_depth_prediction": point_depth_prediction,
                "geometry_delta": geometry_delta,
                "raw": self._masked_trimmed_mean(
                    vote_raw,
                    valid_mask,
                    self.trim_ratio,
                ),
            }


class CenterZResidualHead(nn.Module):
    """Refine an existing object-center depth without changing its image ray.

    The head consumes the preserved V3 global/depth context together with
    inference-available geometry, structured R/S, vote dispersion, category,
    and source identity.  Its final layer is zero initialized, so a fresh V4
    head reproduces the input V3 translation exactly before training.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_sources: int = 2,
        num_classes: int = 6,
        geometry_dim: int = 18,
        structured_pose_dim: int = 9,
        dispersion_dim: int = 6,
        max_log_depth_residual: float = 0.40,
        min_depth: float = 0.10,
        max_depth: float = 3.00,
    ) -> None:
        super().__init__()
        token_dim = int(token_dim)
        self.geometry_dim = int(geometry_dim)
        self.structured_pose_dim = int(structured_pose_dim)
        self.dispersion_dim = int(dispersion_dim)
        self.max_log_depth_residual = float(max_log_depth_residual)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        if int(num_sources) <= 0:
            raise ValueError("center-Z num_sources must be positive")
        if int(num_classes) <= 0:
            raise ValueError("center-Z num_classes must be positive")
        if self.geometry_dim <= 0 or self.structured_pose_dim <= 0:
            raise ValueError("center-Z geometry dimensions must be positive")
        if self.dispersion_dim <= 0:
            raise ValueError("center-Z dispersion_dim must be positive")
        if self.max_log_depth_residual <= 0.0:
            raise ValueError("center-Z log-depth bound must be positive")
        if self.min_depth <= 0.0 or self.max_depth <= self.min_depth:
            raise ValueError("center-Z metric range is invalid")

        self.depth_context_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.global_query = DeterministicPoseQuery(
            token_dim,
            int(num_heads),
            int(num_layers),
            float(dropout),
        )
        self.source_embedding = nn.Embedding(int(num_sources), token_dim)
        self.category_embedding = nn.Embedding(int(num_classes), token_dim)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(token_dim * 4),
            nn.Linear(token_dim * 4, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        # geometry (18), log-size + rotation-6D (9), vote dispersion (6),
        # input/surface/log-ratio depths (3), and bbox-normalized center (2).
        numerical_dim = (
            self.geometry_dim
            + self.structured_pose_dim
            + self.dispersion_dim
            + 5
        )
        self.numerical_projection = nn.Sequential(
            nn.LayerNorm(numerical_dim),
            nn.Linear(numerical_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.residual_output = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, 1),
        )
        nn.init.zeros_(self.residual_output[-1].weight)
        nn.init.zeros_(self.residual_output[-1].bias)

    def forward(
        self,
        depth_token: torch.Tensor,
        context: torch.Tensor,
        source_indices: torch.Tensor,
        category_indices: torch.Tensor,
        geometry_features: torch.Tensor,
        structured_size: torch.Tensor,
        structured_rotation_6d: torch.Tensor,
        vote_dispersion: torch.Tensor,
        input_translation: torch.Tensor,
        surface_depth: torch.Tensor,
        input_center_uv: torch.Tensor,
        bbox_uv: torch.Tensor,
        bbox_wh: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with autocast_disabled(depth_token.device):
            depth_token = self.depth_context_projection(depth_token.float())
            context = context.float()
            geometry_features = geometry_features.float()
            structured_size = structured_size.float()
            structured_rotation_6d = structured_rotation_6d.float()
            vote_dispersion = vote_dispersion.float()
            input_translation = input_translation.float()
            surface_depth = surface_depth.float().view(-1)
            input_center_uv = input_center_uv.float()
            bbox_uv = bbox_uv.float()
            bbox_wh = bbox_wh.float().clamp_min(1.0)

            if geometry_features.shape[-1] != self.geometry_dim:
                raise ValueError(
                    "center-Z geometry feature mismatch: expected {}, got {}".format(
                        self.geometry_dim,
                        geometry_features.shape[-1],
                    )
                )
            structured_pose = torch.cat(
                [
                    torch.log(structured_size.clamp_min(1.0e-6)),
                    structured_rotation_6d,
                ],
                dim=-1,
            )
            if structured_pose.shape[-1] != self.structured_pose_dim:
                raise ValueError(
                    "center-Z structured pose mismatch: expected {}, got {}".format(
                        self.structured_pose_dim,
                        structured_pose.shape[-1],
                    )
                )
            if vote_dispersion.shape[-1] != self.dispersion_dim:
                raise ValueError(
                    "center-Z dispersion mismatch: expected {}, got {}".format(
                        self.dispersion_dim,
                        vote_dispersion.shape[-1],
                    )
                )

            source_indices = source_indices.to(
                device=depth_token.device,
                dtype=torch.long,
            ).view(-1).clamp(0, self.source_embedding.num_embeddings - 1)
            category_indices = category_indices.to(
                device=depth_token.device,
                dtype=torch.long,
            ).view(-1).clamp(0, self.category_embedding.num_embeddings - 1)

            input_depth = input_translation[:, 2].clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            surface_depth = surface_depth.clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            center_offset = (input_center_uv - bbox_uv) / bbox_wh
            depth_features = torch.stack(
                [
                    torch.log(input_depth),
                    torch.log(surface_depth),
                    torch.log(input_depth / surface_depth.clamp_min(1.0e-6)),
                ],
                dim=-1,
            )
            numerical_features = torch.cat(
                [
                    geometry_features,
                    structured_pose,
                    vote_dispersion,
                    depth_features,
                    center_offset,
                ],
                dim=-1,
            )
            numerical_features = torch.nan_to_num(
                numerical_features,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0,
            ).clamp(-10.0, 10.0)

            augmented_context = torch.cat(
                [depth_token.unsqueeze(1), context],
                dim=1,
            )
            global_feature = self.global_query(augmented_context)
            condition = self.condition_projection(
                torch.cat(
                    [
                        global_feature,
                        depth_token,
                        self.source_embedding(source_indices),
                        self.category_embedding(category_indices),
                    ],
                    dim=-1,
                )
            )
            condition = condition + self.numerical_projection(numerical_features)
            raw_residual = self.residual_output(condition).squeeze(-1)
            requested_delta = self.max_log_depth_residual * torch.tanh(raw_residual)
            refined_depth = (input_depth * torch.exp(requested_delta)).clamp(
                min=self.min_depth,
                max=self.max_depth,
            )
            applied_delta = torch.log(
                refined_depth / input_depth.clamp_min(1.0e-6)
            )
            ray_scale = refined_depth / input_depth.clamp_min(1.0e-6)
            refined_translation = input_translation * ray_scale.unsqueeze(-1)
            return {
                "translation": refined_translation,
                "depth": refined_depth,
                "center_uv": input_center_uv,
                "delta_log_depth": applied_delta,
                "raw_delta_log_depth": raw_residual,
                "base_translation": input_translation,
                "base_depth": input_depth,
                "surface_depth": surface_depth,
                # Reused by the guarded V11 wrapper. Returning this feature is
                # parameter-free and leaves every existing V4/V5 checkpoint
                # key unchanged.
                "condition": condition,
            }


class GuardedCenterZV11Head(nn.Module):
    """Predict a bounded Center-Z candidate with an optional learned gate.

    With the gate enabled this retains the original V11 behavior.  With it
    disabled, the candidate is applied directly and only invalid geometry or an
    excessive residual can trigger the hard safety fallback.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_sources: int = 2,
        num_classes: int = 6,
        geometry_dim: int = 18,
        structured_pose_dim: int = 9,
        dispersion_dim: int = 6,
        max_log_depth_residual: float = 0.20,
        max_apply_log_depth_residual: float = 0.18,
        confidence_threshold: float = 0.65,
        confidence_gate_enable: bool = True,
        min_depth: float = 0.10,
        max_depth: float = 3.00,
    ) -> None:
        super().__init__()
        self.max_apply_log_depth_residual = float(max_apply_log_depth_residual)
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_gate_enable = bool(confidence_gate_enable)
        if not 0.5 < self.confidence_threshold < 1.0:
            raise ValueError("V11 confidence_threshold must be in (0.5, 1.0)")
        if not 0.0 < self.max_apply_log_depth_residual <= float(
            max_log_depth_residual
        ):
            raise ValueError(
                "V11 max_apply_log_depth_residual must be positive and no "
                "larger than max_log_depth_residual"
            )

        self.candidate_head = CenterZResidualHead(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_sources=num_sources,
            num_classes=num_classes,
            geometry_dim=geometry_dim,
            structured_pose_dim=structured_pose_dim,
            dispersion_dim=dispersion_dim,
            max_log_depth_residual=max_log_depth_residual,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        self.confidence_output = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.SiLU(),
            nn.Linear(int(token_dim), 1),
        )
        nn.init.zeros_(self.confidence_output[-1].weight)
        nn.init.zeros_(self.confidence_output[-1].bias)

    def forward(
        self,
        *,
        geometry_valid: torch.Tensor,
        **center_z_inputs,
    ) -> dict[str, torch.Tensor]:
        candidate = self.candidate_head(**center_z_inputs)
        base_translation = center_z_inputs["input_translation"].float()
        candidate_translation = candidate["translation"]
        delta_log_depth = candidate["delta_log_depth"]
        if self.confidence_gate_enable:
            improve_logit = self.confidence_output(candidate["condition"]).squeeze(-1)
            confidence = torch.sigmoid(improve_logit)
        else:
            # Keep the output contract stable for checkpoint/test tooling while
            # removing the learned gate from the actual inference path.
            improve_logit = torch.full_like(delta_log_depth, 20.0)
            confidence = torch.ones_like(delta_log_depth)

        geometry_valid = geometry_valid.to(
            device=base_translation.device,
            dtype=torch.bool,
        ).view(-1)
        invalid_mask = (
            ~geometry_valid
            | ~torch.isfinite(base_translation).all(dim=1)
            | ~torch.isfinite(candidate_translation).all(dim=1)
            | ~torch.isfinite(delta_log_depth)
            | ~torch.isfinite(confidence)
            | (base_translation[:, 2] <= 1.0e-6)
            | (candidate_translation[:, 2] <= 1.0e-6)
        )
        low_confidence_mask = (
            confidence < self.confidence_threshold
            if self.confidence_gate_enable
            else torch.zeros_like(invalid_mask)
        )
        excessive_residual_mask = (
            delta_log_depth.abs() > self.max_apply_log_depth_residual
        )
        fallback_mask = (
            invalid_mask | low_confidence_mask | excessive_residual_mask
        )
        final_translation = torch.where(
            fallback_mask.unsqueeze(-1),
            base_translation,
            candidate_translation,
        )

        return {
            "translation": final_translation,
            "candidate_translation": candidate_translation,
            "base_translation": base_translation,
            "depth": final_translation[:, 2],
            "candidate_depth": candidate["depth"],
            "base_depth": base_translation[:, 2],
            "surface_depth": candidate["surface_depth"],
            "center_uv": candidate["center_uv"],
            "delta_log_depth": delta_log_depth,
            "raw_delta_log_depth": candidate["raw_delta_log_depth"],
            "improve_logit": improve_logit,
            "confidence": confidence,
            "fallback_mask": fallback_mask,
            "invalid_mask": invalid_mask,
            "low_confidence_mask": low_confidence_mask,
            "excessive_residual_mask": excessive_residual_mask,
        }


class PositiveSizeHead(nn.Module):
    def __init__(self, token_dim: int, min_size: float = 1.0e-4, max_size: float = 2.0):
        super().__init__()
        self.min_size = float(min_size)
        self.max_size = float(max_size)
        self.head = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim) * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(int(token_dim) * 2, 3),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        size = F.softplus(self.head(feature)) + self.min_size
        return size.clamp(max=self.max_size)


class DirectRotationHead(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim) * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(int(token_dim) * 2, 6),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        raw = self.head(feature)
        first = F.normalize(raw[:, :3], dim=-1, eps=1.0e-6)
        second = raw[:, 3:] - (first * raw[:, 3:]).sum(dim=-1, keepdim=True) * first
        second = F.normalize(second, dim=-1, eps=1.0e-6)
        return torch.cat([first, second], dim=-1)


class StructuredRSHead(nn.Module):
    """Fresh full-context deterministic rotation and metric-size predictor."""

    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        num_heads: int,
        context_layers: int,
        query_layers: int,
        dropout: float,
        min_size: float = 1.0e-4,
        max_size: float = 2.0,
        initial_size: float = 0.20,
    ) -> None:
        super().__init__()
        self.context_encoder = StaticPoseContextEncoder(
            token_dim=token_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=context_layers,
            dropout=dropout,
        )
        self.rotation_query = DeterministicPoseQuery(
            token_dim, num_heads, query_layers, dropout
        )
        self.size_query = DeterministicPoseQuery(
            token_dim, num_heads, query_layers, dropout
        )
        self.rotation_head = DirectRotationHead(token_dim)
        self.size_head = PositiveSizeHead(
            token_dim=token_dim,
            min_size=min_size,
            max_size=max_size,
        )

        # Start from a valid identity-like rotation and a realistic positive
        # size without copying either legacy R/S head. Tiny non-zero weights let
        # gradients reach the query stacks on the first optimization step.
        rotation_last = self.rotation_head.head[-1]
        nn.init.normal_(rotation_last.weight, std=1.0e-3)
        with torch.no_grad():
            rotation_last.bias.copy_(
                rotation_last.bias.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )
        size_last = self.size_head.head[-1]
        nn.init.normal_(size_last.weight, std=1.0e-3)
        target = max(float(initial_size) - float(min_size), 1.0e-4)
        inverse_softplus = math.log(math.expm1(target))
        nn.init.constant_(size_last.bias, inverse_softplus)

    def forward(
        self,
        da_global: torch.Tensor,
        octree_token: torch.Tensor,
        fused_token: torch.Tensor,
        category: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # This branch is optimized with geometric rotation losses whose
        # normalization and acos derivatives are unsafe in reduced precision.
        # Cast the frozen feature inputs once and keep the complete fresh R/S
        # head, including the 6D-to-SO(3) projection, outside autocast.
        with autocast_disabled(da_global.device):
            context = self.context_encoder(
                da_global=da_global.float(),
                octree_token=octree_token.float(),
                fused_token=fused_token.float(),
                category=category,
                spatial_tokens=spatial_tokens.float(),
            )
            rotation_feature = self.rotation_query(context)
            size_feature = self.size_query(context)
            rotation_6d = self.rotation_head(rotation_feature)
            size = self.size_head(size_feature)
            rotation = six_d_to_rotation_matrix(rotation_6d)
        return {
            "context": context,
            "rotation_6d": rotation_6d,
            "rotation": rotation,
            "size": size,
        }


class RotationV6ResidualRefiner(nn.Module):
    """Baseline-conditioned multi-hypothesis local SO(3) refiner.

    Hypothesis zero is an immutable identity fallback.  Every learned residual
    head has an exactly zero-initialized final layer, and all selection logits
    start at zero, so an untrained V6 route is bitwise-equivalent to the frozen
    structured V4 rotation.  The learned hypotheses read both the frozen pose
    context and points canonicalized by the V4 pose.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_hypotheses: int = 4,
        geometry_dim: int = 18,
        max_residual_degrees: float = 30.0,
    ) -> None:
        super().__init__()
        token_dim = int(token_dim)
        self.num_hypotheses = int(num_hypotheses)
        if self.num_hypotheses < 2:
            raise ValueError("rotation_v6.num_hypotheses must be at least 2")
        self.num_learned_hypotheses = self.num_hypotheses - 1
        self.max_residual_radians = math.radians(float(max_residual_degrees))
        if self.max_residual_radians <= 0.0:
            raise ValueError("rotation_v6.max_residual_degrees must be positive")

        point_hidden = max(token_dim // 2, 64)
        self.point_encoder = nn.Sequential(
            nn.Linear(4, point_hidden),
            nn.LayerNorm(point_hidden),
            nn.SiLU(),
            nn.Linear(point_hidden, point_hidden),
            nn.LayerNorm(point_hidden),
            nn.SiLU(),
        )
        self.point_pool_projection = nn.Sequential(
            nn.Linear(point_hidden * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.base_geometry_projection = nn.Sequential(
            nn.LayerNorm(6 + 3 + int(geometry_dim)),
            nn.Linear(6 + 3 + int(geometry_dim), token_dim * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.hypothesis_queries = nn.Parameter(
            torch.zeros(1, self.num_learned_hypotheses, token_dim)
        )
        self.blocks = nn.ModuleList(
            [
                PoseCrossAttentionBlock(token_dim, int(num_heads), float(dropout))
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.residual_heads = nn.ModuleList()
        for _ in range(self.num_learned_hypotheses):
            head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, token_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(token_dim, 3),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.residual_heads.append(head)
        self.identity_score = nn.Linear(token_dim, 1)
        self.correction_scores = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.identity_score.weight)
        nn.init.zeros_(self.identity_score.bias)
        nn.init.zeros_(self.correction_scores.weight)
        nn.init.zeros_(self.correction_scores.bias)
        nn.init.normal_(self.hypothesis_queries, std=0.02)

    @staticmethod
    def _masked_point_pool(
        point_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask.to(device=point_features.device).bool()
        valid_f = valid.to(dtype=point_features.dtype).unsqueeze(-1)
        count = valid_f.sum(dim=1).clamp_min(1.0)
        mean = (point_features * valid_f).sum(dim=1) / count
        masked = point_features.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        maximum = masked.amax(dim=1)
        maximum = torch.where(
            valid.any(dim=1, keepdim=True),
            maximum,
            torch.zeros_like(maximum),
        )
        return torch.cat([mean, maximum], dim=-1)

    def _bound_residual(self, raw: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        scale = self.max_residual_radians / (
            self.max_residual_radians + norm
        )
        return raw * scale

    def forward(
        self,
        context: torch.Tensor,
        canonical_points: torch.Tensor,
        point_valid: torch.Tensor,
        base_rotation: torch.Tensor,
        base_size: torch.Tensor,
        geometry_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with autocast_disabled(context.device):
            context = context.float()
            canonical_points = torch.nan_to_num(
                canonical_points.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(-4.0, 4.0)
            point_valid = point_valid.to(device=context.device).bool()
            radius = torch.linalg.vector_norm(
                canonical_points, dim=-1, keepdim=True
            )
            encoded_points = self.point_encoder(
                torch.cat([canonical_points, radius], dim=-1)
            )
            point_token = self.point_pool_projection(
                self._masked_point_pool(encoded_points, point_valid)
            )

            base_rotation = base_rotation.float()
            base_size = base_size.float().clamp_min(1.0e-6)
            geometry_features = torch.nan_to_num(
                geometry_features.float(), nan=0.0, posinf=10.0, neginf=-10.0
            ).clamp(-10.0, 10.0)
            base_condition = self.base_geometry_projection(
                torch.cat(
                    [
                        rotation_matrix_to_6d(base_rotation),
                        torch.log(base_size),
                        geometry_features,
                    ],
                    dim=-1,
                )
            )
            condition = base_condition + point_token
            query = self.hypothesis_queries.expand(context.shape[0], -1, -1)
            query = query + condition.unsqueeze(1)
            augmented_context = torch.cat(
                [context, point_token.unsqueeze(1), base_condition.unsqueeze(1)],
                dim=1,
            )
            for block in self.blocks:
                query = block(query, augmented_context)
            query = self.output_norm(query)

            learned_residuals = []
            for index, head in enumerate(self.residual_heads):
                learned_residuals.append(head(query[:, index]))
            learned_residuals = self._bound_residual(
                torch.stack(learned_residuals, dim=1)
            )
            identity_residual = learned_residuals.new_zeros(
                learned_residuals.shape[0], 1, 3
            )
            residuals = torch.cat([identity_residual, learned_residuals], dim=1)
            residual_rotations = axis_angle_to_rotation_matrix(residuals)
            hypotheses = torch.matmul(
                residual_rotations,
                base_rotation.unsqueeze(1),
            )

            identity_logit = self.identity_score(condition)
            correction_logits = self.correction_scores(query).squeeze(-1)
            logits = torch.cat([identity_logit, correction_logits], dim=1)
            selected_index = logits.argmax(dim=1)
            batch_indices = torch.arange(
                hypotheses.shape[0], device=hypotheses.device
            )
            selected_rotation = hypotheses[batch_indices, selected_index]

        return {
            "rotation": selected_rotation,
            "base_rotation": base_rotation,
            "hypotheses": hypotheses,
            "axis_angle_residuals": residuals,
            "selection_logits": logits,
            "selected_index": selected_index,
            "point_token": point_token,
        }


class TranslationAnchorHead(nn.Module):
    def __init__(self, token_dim: int, max_offset: float = 0.5):
        super().__init__()
        self.max_offset = float(max_offset)
        self.head = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim) * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(int(token_dim) * 2, 3),
        )

    def forward(self, feature: torch.Tensor, point_center: torch.Tensor) -> torch.Tensor:
        offset = self.max_offset * torch.tanh(self.head(feature))
        return point_center + offset


class PointNOCSHead(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(9, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim), 4),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.point_encoder(point_features)
        global_token = self.global_projection(global_feature).unsqueeze(1)
        global_token = global_token.expand(-1, local.shape[1], -1)
        output = self.output(torch.cat([local, global_token], dim=-1))
        return 0.6 * torch.tanh(output[..., :3]), output[..., 3]


class RotationCorrespondenceV9(nn.Module):
    """Category-aware absolute rotation correspondences over frozen features.

    The module predicts one canonical coordinate and one reliability logit for
    every metric point.  It deliberately does not consume a baseline rotation:
    the caller recovers an absolute SO(3) estimate with weighted Kabsch.  RGB
    tokens, camera-space points, and auxiliary point attributes are detached by
    the model boundary before they enter this branch, so only this module is
    trainable in the isolated V9 experiment.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 6,
        dropout: float = 0.1,
        nocs_limit: float = 0.6,
    ) -> None:
        super().__init__()
        token_dim = int(token_dim)
        hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.nocs_limit = float(nocs_limit)
        if hidden_dim <= 0:
            raise ValueError("rotation_correspondence_v9.hidden_dim must be positive")
        if self.num_classes <= 0:
            raise ValueError("rotation_correspondence_v9.num_classes must be positive")
        if self.nocs_limit <= 0.0:
            raise ValueError("rotation_correspondence_v9.nocs_limit must be positive")

        # normalized camera xyz + RGB + normals + sampled DA2 token
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(token_dim + 9),
            nn.Linear(token_dim + 9, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
        )
        self.category_embedding = nn.Embedding(self.num_classes, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # Category-local outputs avoid forcing symmetric and asymmetric object
        # families to share the last correspondence mapping.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, 4),
                )
                for _ in range(self.num_classes)
            ]
        )

    @staticmethod
    def _sample_spatial_tokens(
        spatial_tokens: torch.Tensor,
        choose: torch.Tensor,
        bbox: torch.Tensor,
    ) -> torch.Tensor:
        if spatial_tokens.ndim != 3:
            raise ValueError("spatial_tokens must have shape [B, G*G, C]")
        if choose.ndim != 3 or choose.shape[-1] != 2:
            raise ValueError("choose must have shape [B, N, 2] in row/column order")
        if bbox.ndim != 2 or bbox.shape[-1] != 4:
            raise ValueError("bbox must have shape [B, 4] in y1/x1/y2/x2 order")

        token_count = int(spatial_tokens.shape[1])
        grid_size = int(round(math.sqrt(token_count)))
        if grid_size * grid_size != token_count:
            raise ValueError(
                f"V9 needs a square spatial-token grid, got {token_count} tokens"
            )
        token_grid = spatial_tokens.transpose(1, 2).reshape(
            spatial_tokens.shape[0],
            spatial_tokens.shape[2],
            grid_size,
            grid_size,
        )

        bbox = bbox.to(device=spatial_tokens.device, dtype=spatial_tokens.dtype)
        choose = choose.to(device=spatial_tokens.device, dtype=spatial_tokens.dtype)
        crop_height = (bbox[:, 2] - bbox[:, 0]).clamp_min(1.0).unsqueeze(1)
        crop_width = (bbox[:, 3] - bbox[:, 1]).clamp_min(1.0).unsqueeze(1)
        # choose contains pixels in the unresized object crop.  Pixel-centre
        # normalization matches grid_sample(..., align_corners=False).
        grid_x = 2.0 * (choose[..., 1] + 0.5) / crop_width - 1.0
        grid_y = 2.0 * (choose[..., 0] + 0.5) / crop_height - 1.0
        sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            token_grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    @staticmethod
    def _masked_point_statistics(
        features: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_f = valid.to(dtype=features.dtype).unsqueeze(-1)
        count = valid_f.sum(dim=1).clamp_min(1.0)
        mean = (features * valid_f).sum(dim=1) / count
        masked = features.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        maximum = masked.amax(dim=1)
        maximum = torch.where(
            valid.any(dim=1, keepdim=True),
            maximum,
            torch.zeros_like(maximum),
        )
        return mean, maximum

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        global_feature: torch.Tensor,
        metric_points: torch.Tensor,
        point_valid: torch.Tensor,
        choose: torch.Tensor,
        bbox: torch.Tensor,
        rgb_points: torch.Tensor,
        normals: torch.Tensor,
        category: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with autocast_disabled(spatial_tokens.device):
            spatial_tokens = spatial_tokens.float()
            global_feature = global_feature.float()
            metric_points = torch.nan_to_num(
                metric_points.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            point_valid = point_valid.to(device=metric_points.device).bool()
            point_valid = point_valid & torch.isfinite(metric_points).all(dim=-1)
            rgb_points = torch.nan_to_num(
                rgb_points.float(), nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
            normals = torch.nan_to_num(
                normals.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(-1.0, 1.0)

            valid_f = point_valid.to(dtype=metric_points.dtype).unsqueeze(-1)
            count = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            center = (metric_points * valid_f).sum(dim=1, keepdim=True) / count
            centered = metric_points - center
            radius = torch.linalg.vector_norm(centered, dim=-1)
            radius = radius.masked_fill(~point_valid, 0.0)
            extent = radius.amax(dim=1, keepdim=True).clamp_min(1.0e-4)
            normalized_points = centered / extent.unsqueeze(-1)
            normalized_points = torch.where(
                point_valid.unsqueeze(-1),
                normalized_points,
                torch.zeros_like(normalized_points),
            )

            sampled_tokens = self._sample_spatial_tokens(
                spatial_tokens,
                choose,
                bbox,
            )
            local = self.point_encoder(
                torch.cat(
                    [normalized_points, rgb_points, normals, sampled_tokens],
                    dim=-1,
                )
            )
            local = torch.where(
                point_valid.unsqueeze(-1),
                local,
                torch.zeros_like(local),
            )
            pooled_mean, pooled_max = self._masked_point_statistics(
                local,
                point_valid,
            )
            global_token = self.global_projection(global_feature)
            category = category.to(device=local.device).view(-1).long()
            category = category.clamp(0, self.num_classes - 1)
            category_token = self.category_embedding(category)
            condition = torch.cat(
                [pooled_mean, pooled_max, global_token, category_token],
                dim=-1,
            )
            fused = self.fusion(condition).unsqueeze(1) + local

            expert_outputs = torch.stack(
                [expert(fused) for expert in self.experts],
                dim=2,
            )
            gather_index = category.view(-1, 1, 1, 1).expand(
                -1,
                fused.shape[1],
                1,
                4,
            )
            selected = expert_outputs.gather(2, gather_index).squeeze(2)
            pred_nocs = self.nocs_limit * torch.tanh(selected[..., :3])
            confidence_logits = selected[..., 3]
            pred_nocs = torch.where(
                point_valid.unsqueeze(-1),
                pred_nocs,
                torch.zeros_like(pred_nocs),
            )
            confidence_logits = torch.where(
                point_valid,
                confidence_logits,
                confidence_logits.new_full(confidence_logits.shape, -12.0),
            )
        return {
            "nocs": pred_nocs,
            "confidence_logits": confidence_logits,
            "confidence": torch.sigmoid(confidence_logits),
            "point_valid": point_valid,
            "sampled_visual_tokens": sampled_tokens,
        }


def fake_backbone_output(feature_maps):
    """Small helper used by unit tests without importing Transformers."""
    return SimpleNamespace(feature_maps=feature_maps)
