from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.da2_e2e_pose import DA2E2EPose, _select
from utils.amp_utils import autocast_disabled


class FrozenDA2MetricDepth(nn.Module):
    """Frozen DA2 Metric Indoor inference without writing depth images."""

    def __init__(
        self,
        model_name: str,
        input_size: int = 518,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("metric_center_z.input_size must be positive")

        if model is None:
            try:
                from transformers import AutoConfig, AutoModelForDepthEstimation
            except ImportError as exc:
                raise ImportError(
                    "DA2 Metric Center-Z requires transformers"
                ) from exc
            config = AutoConfig.from_pretrained(
                self.model_name,
                local_files_only=bool(local_files_only),
                revision=revision,
            )
            # Some training machines retain an older relative-depth config in
            # the HF cache. That selects ReLU and can turn the metric checkpoint
            # into an all-zero predictor. The selected repository is explicitly
            # metric, so make that inference contract independent of the cache.
            config.depth_estimation_type = "metric"
            model = AutoModelForDepthEstimation.from_pretrained(
                self.model_name,
                config=config,
                local_files_only=bool(local_files_only),
                revision=revision,
                attn_implementation="eager",
                torch_dtype=torch.float32,
            )
        model.float()
        self.model = model
        self.patch_size = int(getattr(getattr(model, "config", None), "patch_size", 14))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        del mode
        return super().train(False)

    def forward(self, scene_rgb: torch.Tensor) -> torch.Tensor:
        if scene_rgb.ndim != 4 or scene_rgb.shape[1] != 3:
            raise ValueError("scene_rgb must have shape [B,3,H,W]")
        output_size = scene_rgb.shape[-2:]
        input_height, input_width = output_size
        scale_height = self.input_size / input_height
        scale_width = self.input_size / input_width
        scale = (
            scale_width
            if abs(1.0 - scale_width) < abs(1.0 - scale_height)
            else scale_height
        )
        resized_height = max(
            self.patch_size,
            round(scale * input_height / self.patch_size) * self.patch_size,
        )
        resized_width = max(
            self.patch_size,
            round(scale * input_width / self.patch_size) * self.patch_size,
        )
        # Metric depth is used as a hard routing signal. Keep it in FP32 even
        # when the surrounding pose-training step runs under AMP.
        with autocast_disabled(scene_rgb.device):
            pixel_values = F.interpolate(
                scene_rgb.float(),
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
            )
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
        if isinstance(outputs, dict):
            predicted_depth = outputs.get("predicted_depth")
        else:
            predicted_depth = getattr(outputs, "predicted_depth", None)
        if predicted_depth is None and isinstance(outputs, (tuple, list)):
            predicted_depth = outputs[0]
        if predicted_depth is None:
            raise ValueError("DA2 metric-depth model did not return predicted_depth")
        if predicted_depth.ndim == 4 and predicted_depth.shape[1] == 1:
            predicted_depth = predicted_depth[:, 0]
        if predicted_depth.ndim != 3:
            raise ValueError(
                "predicted_depth must have shape [B,H,W], got "
                f"{tuple(predicted_depth.shape)}"
            )
        if not torch.isfinite(predicted_depth).any() or predicted_depth.max() <= 0.0:
            config = getattr(self.model, "config", None)
            head = getattr(self.model, "head", None)
            activation = getattr(head, "activation2", None)
            parameter = next(self.model.parameters(), None)
            raise RuntimeError(
                "DA2 metric-depth returned no positive finite values before "
                "resize. model={!r}, depth_type={!r}, max_depth={!r}, "
                "activation={}, parameter_dtype={}, input_dtype={}, "
                "input_range=[{:.6g}, {:.6g}]. Verify that transformers==5.5.0 "
                "is installed and clear only this model's HF cache if the "
                "pinned revision was previously corrupted.".format(
                    self.model_name,
                    getattr(config, "depth_estimation_type", None),
                    getattr(config, "max_depth", None),
                    type(activation).__name__ if activation is not None else "unknown",
                    parameter.dtype if parameter is not None else "unknown",
                    pixel_values.dtype,
                    float(pixel_values.min()),
                    float(pixel_values.max()),
                )
            )
        return F.interpolate(
            predicted_depth.float().unsqueeze(1),
            size=output_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)


class MetricCenterZHead(nn.Module):
    """Class-aware bounded log-depth residual with identity initialization."""

    def __init__(
        self,
        num_classes: int,
        numeric_dim: int,
        category_dim: int,
        hidden_dim: int,
        dropout: float,
        max_log_residual: float,
    ) -> None:
        super().__init__()
        self.max_log_residual = float(max_log_residual)
        if self.max_log_residual <= 0.0:
            raise ValueError("metric_center_z.max_log_residual must be positive")
        self.category_embedding = nn.Embedding(num_classes, category_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(numeric_dim + category_dim),
            nn.Linear(numeric_dim + category_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        numeric_features: torch.Tensor,
        category: torch.Tensor,
    ) -> torch.Tensor:
        feature = torch.cat(
            [numeric_features, self.category_embedding(category)], dim=1
        )
        raw_residual = self.network(feature).squeeze(1)
        return self.max_log_residual * torch.tanh(raw_residual)


class DA2MetricCenterZPose(DA2E2EPose):
    """Frozen E2E pose model with a metric-depth Center-Z calibrator."""

    def __init__(
        self,
        cfg: Any,
        visual_encoder: Optional[nn.Module] = None,
        metric_depth_predictor: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(cfg, visual_encoder=visual_encoder)
        self.metric_center_z_trainable_scope = str(
            _select(cfg, "metric_center_z.trainable_scope", "head_only")
        ).lower()
        valid_trainable_scopes = {"head_only", "pose_and_head"}
        if self.metric_center_z_trainable_scope not in valid_trainable_scopes:
            raise ValueError(
                "metric_center_z.trainable_scope must be one of "
                f"{sorted(valid_trainable_scopes)}, got "
                f"{self.metric_center_z_trainable_scope!r}"
            )
        self.metric_center_z_detach_base_translation = bool(
            _select(cfg, "metric_center_z.detach_base_translation", True)
        )
        self._base_trainable_parameter_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        self.metric_depth_min = float(
            _select(cfg, "metric_center_z.valid_depth_min_m", 0.05)
        )
        self.metric_depth_max = float(
            _select(cfg, "metric_center_z.valid_depth_max_m", 5.0)
        )
        self.metric_min_valid_pixels = int(
            _select(cfg, "metric_center_z.min_valid_pixels", 64)
        )
        self.metric_depth_source = str(
            _select(cfg, "metric_center_z.depth_source", "online_hf")
        ).lower()
        valid_sources = {"online_hf", "dinov2_base", "dinov2_large"}
        if self.metric_depth_source not in valid_sources:
            raise ValueError(
                "metric_center_z.depth_source must be one of "
                f"{sorted(valid_sources)}, got {self.metric_depth_source!r}"
            )
        if not 0.0 < self.metric_depth_min < self.metric_depth_max:
            raise ValueError("metric_center_z valid depth range is invalid")
        if self.metric_min_valid_pixels <= 0:
            raise ValueError("metric_center_z.min_valid_pixels must be positive")

        enabled_classes = [
            int(value)
            for value in _select(cfg, "metric_center_z.enabled_classes", [0, 1])
        ]
        if any(value < 0 or value >= self.num_classes for value in enabled_classes):
            raise ValueError("metric_center_z.enabled_classes contains an invalid class")
        enabled_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        enabled_mask[enabled_classes] = True
        self.register_buffer("metric_center_z_enabled_mask", enabled_mask)

        if metric_depth_predictor is None and self.metric_depth_source == "online_hf":
            metric_depth_predictor = FrozenDA2MetricDepth(
                model_name=str(
                    _select(
                        cfg,
                        "metric_center_z.model_name",
                        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                    )
                ),
                input_size=int(_select(cfg, "metric_center_z.input_size", 518)),
                local_files_only=bool(
                    _select(cfg, "metric_center_z.local_files_only", False)
                ),
                revision=_select(cfg, "metric_center_z.revision", None),
            )
        self.metric_depth_predictor = metric_depth_predictor
        self.metric_center_z_head = MetricCenterZHead(
            num_classes=self.num_classes,
            numeric_dim=14,
            category_dim=int(_select(cfg, "metric_center_z.category_dim", 16)),
            hidden_dim=int(_select(cfg, "metric_center_z.hidden_dim", 128)),
            dropout=float(_select(cfg, "metric_center_z.dropout", 0.10)),
            max_log_residual=float(
                _select(cfg, "metric_center_z.max_log_residual", 0.50)
            ),
        )
        self._freeze_for_metric_center_z()

    def _freeze_for_metric_center_z(self) -> None:
        self.requires_grad_(False)
        if self.metric_center_z_trainable_scope == "pose_and_head":
            for name, parameter in self.named_parameters():
                if name in self._base_trainable_parameter_names:
                    parameter.requires_grad_(True)
        self.metric_center_z_head.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.children():
            if module is self.metric_center_z_head:
                module.train(mode)
            elif (
                self.metric_center_z_trainable_scope == "pose_and_head"
                and module is not self.metric_depth_predictor
                and any(parameter.requires_grad for parameter in module.parameters())
            ):
                module.train(mode)
            else:
                module.eval()
        return self

    @staticmethod
    def _scene_inputs(
        inputs: dict[str, torch.Tensor],
        object_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scene_rgb = inputs.get("scene_rgb")
        object_mask = inputs.get("scene_object_mask")
        if scene_rgb is None or object_mask is None:
            raise KeyError(
                "DA2MetricCenterZPose requires scene_rgb and scene_object_mask"
            )
        if scene_rgb.ndim == 3:
            scene_rgb = scene_rgb.unsqueeze(0)
        if object_mask.ndim == 2:
            object_mask = object_mask.unsqueeze(0)
        if scene_rgb.ndim != 4 or scene_rgb.shape[1] != 3:
            raise ValueError("scene_rgb must have shape [M,3,H,W]")
        if object_mask.ndim != 3 or object_mask.shape[0] != object_count:
            raise ValueError(
                "scene_object_mask must have shape [object_count,H,W]"
            )
        if scene_rgb.shape[0] not in (1, object_count):
            raise ValueError(
                "scene_rgb batch must contain one shared scene or one scene per object"
            )
        return scene_rgb, object_mask.bool()

    def _masked_depth_statistics(
        self,
        depth_maps: torch.Tensor,
        object_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        object_count = object_masks.shape[0]
        if depth_maps.shape[0] not in (1, object_count):
            raise ValueError("metric depth batch does not match object masks")
        if object_masks.shape[-2:] != depth_maps.shape[-2:]:
            object_masks = F.interpolate(
                object_masks.float().unsqueeze(1),
                size=depth_maps.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()

        map_indices = (
            torch.zeros(object_count, device=depth_maps.device, dtype=torch.long)
            if depth_maps.shape[0] == 1
            else torch.arange(object_count, device=depth_maps.device)
        )
        selected_depth = depth_maps[map_indices].float()
        object_masks = object_masks.to(device=selected_depth.device)
        finite_pixels = object_masks & torch.isfinite(selected_depth)
        valid_pixels = (
            finite_pixels
            & (selected_depth >= self.metric_depth_min)
            & (selected_depth <= self.metric_depth_max)
        )

        q25 = selected_depth.new_zeros(object_count)
        median = selected_depth.new_zeros(object_count)
        q75 = selected_depth.new_zeros(object_count)
        depth_min = selected_depth.new_full((object_count,), float("nan"))
        depth_max = selected_depth.new_full((object_count,), float("nan"))
        valid_count = valid_pixels.flatten(1).sum(dim=1)
        finite_count = finite_pixels.flatten(1).sum(dim=1)
        mask_count = object_masks.flatten(1).sum(dim=1)
        valid = valid_count >= self.metric_min_valid_pixels
        for index in range(object_count):
            if finite_count[index] > 0:
                finite_values = selected_depth[index][finite_pixels[index]]
                depth_min[index] = finite_values.min()
                depth_max[index] = finite_values.max()
            if valid[index]:
                quantiles = torch.quantile(
                    selected_depth[index][valid_pixels[index]],
                    selected_depth.new_tensor([0.25, 0.50, 0.75]),
                )
                q25[index], median[index], q75[index] = quantiles.unbind()

        total_pixels = float(selected_depth.shape[-2] * selected_depth.shape[-1])
        return {
            "q25": q25,
            "median": median,
            "q75": q75,
            "depth_min": depth_min,
            "depth_max": depth_max,
            "mask_count": mask_count,
            "finite_count": finite_count,
            "valid_count": valid_count,
            "valid_ratio": valid_count.float() / mask_count.clamp_min(1).float(),
            "mask_ratio": mask_count.float() / total_pixels,
            "valid": valid,
        }

    def _numeric_features(
        self,
        inputs: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        stats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        translation = outputs["pred_translation"].detach().float()
        rotation = outputs["pred_rotation"].detach().float()
        size = outputs["pred_size"].detach().float().clamp_min(1.0e-4)
        current_z = translation[:, 2].clamp_min(1.0e-4)
        metric_median = torch.where(
            stats["valid"], stats["median"], current_z
        ).clamp_min(1.0e-4)

        bbox = self._bbox(inputs, torch.float32).to(current_z.device)
        y1, x1, y2, x2 = bbox.unbind(dim=1)
        image_hw = inputs.get("image_hw")
        if image_hw is None:
            scene_hw = inputs["scene_object_mask"].shape[-2:]
            image_hw = current_z.new_tensor(scene_hw).expand(current_z.shape[0], -1)
        else:
            image_hw = image_hw.to(current_z.device, torch.float32).reshape(-1, 2)
            if image_hw.shape[0] == 1 and current_z.shape[0] > 1:
                image_hw = image_hw.expand(current_z.shape[0], -1)
        image_h, image_w = image_hw.unbind(dim=1)
        bbox_w_ratio = (x2 - x1).clamp_min(1.0) / image_w.clamp_min(1.0)
        bbox_h_ratio = (y2 - y1).clamp_min(1.0) / image_h.clamp_min(1.0)

        mask_area_ratio = inputs.get("mask_area_ratio")
        if mask_area_ratio is None:
            mask_area_ratio = stats["mask_ratio"] / (
                bbox_w_ratio * bbox_h_ratio
            ).clamp_min(1.0e-4)
        else:
            mask_area_ratio = mask_area_ratio.to(
                current_z.device, torch.float32
            ).reshape(-1)

        camera_depth_extent = 0.5 * (
            rotation[:, 2, :].abs() * size
        ).sum(dim=1)
        current_surface_z = outputs.get("pred_e2e_surface_depth_z", current_z)
        current_surface_z = current_surface_z.detach().float().clamp_min(1.0e-4)
        relative_iqr = (stats["q75"] - stats["q25"]).clamp_min(0.0) / metric_median

        return torch.stack(
            [
                torch.log(current_z),
                torch.log(metric_median),
                torch.log(metric_median / current_z),
                relative_iqr,
                stats["valid_ratio"],
                stats["mask_ratio"],
                bbox_w_ratio,
                bbox_h_ratio,
                mask_area_ratio,
                torch.log(size[:, 0]),
                torch.log(size[:, 1]),
                torch.log(size[:, 2]),
                camera_depth_extent / current_z,
                torch.log(current_surface_z / current_z),
            ],
            dim=1,
        )

    def _metric_depth(
        self,
        inputs: dict[str, torch.Tensor],
        scene_rgb: torch.Tensor,
    ) -> torch.Tensor:
        if self.metric_depth_predictor is not None:
            return self.metric_depth_predictor(scene_rgb)

        metric_depth = inputs.get("scene_metric_depth")
        if metric_depth is None:
            raise KeyError(
                "Precomputed Metric Center-Z requires scene_metric_depth. "
                "Enable train/test metric_depth with the same dinov2 source."
            )
        if metric_depth.ndim == 2:
            metric_depth = metric_depth.unsqueeze(0)
        if metric_depth.ndim == 4 and metric_depth.shape[1] == 1:
            metric_depth = metric_depth[:, 0]
        if metric_depth.ndim != 3:
            raise ValueError(
                "scene_metric_depth must have shape [B,H,W], got "
                f"{tuple(metric_depth.shape)}"
            )
        return metric_depth.to(device=scene_rgb.device, dtype=torch.float32)

    def calibrate_pose_outputs(
        self,
        inputs: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        category: Optional[torch.Tensor] = None,
        metric_depth: Optional[torch.Tensor] = None,
        stats: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Apply metric Center-Z calibration to already decoded pose outputs."""
        object_count = outputs["pred_translation"].shape[0]
        if stats is None:
            scene_rgb, object_masks = self._scene_inputs(inputs, object_count)
            if metric_depth is None:
                metric_depth = self._metric_depth(inputs, scene_rgb)
            stats = self._masked_depth_statistics(metric_depth, object_masks)
        if category is None:
            category = self._category_indices(inputs, object_count)
        category = category.to(outputs["pred_translation"].device).reshape(-1).long()
        if category.shape[0] != object_count:
            raise ValueError("category hypothesis count must match pose output count")
        numeric_features = self._numeric_features(inputs, outputs, stats)
        delta_log_z = self.metric_center_z_head(numeric_features, category)

        enabled = self.metric_center_z_enabled_mask[category]
        route = enabled & stats["valid"]
        applied_delta = torch.where(route, delta_log_z, torch.zeros_like(delta_log_z))
        raw_base_translation = outputs["pred_translation"]
        base_translation = (
            raw_base_translation.detach()
            if self.metric_center_z_detach_base_translation
            else raw_base_translation
        )
        base_z = base_translation[:, 2].clamp_min(1.0e-6)
        calibrated_z = (base_z * torch.exp(applied_delta.to(base_z.dtype))).clamp(
            min=self.depth_min,
            max=self.depth_max,
        )
        calibrated_translation = base_translation * (calibrated_z / base_z).unsqueeze(1)

        outputs.update(
            {
                "pred_translation_base": raw_base_translation.detach(),
                "pred_translation": calibrated_translation,
                "pred_metric_depth_z": calibrated_z,
                "pred_metric_center_z_surface": stats["median"],
                "pred_metric_center_z_q25": stats["q25"],
                "pred_metric_center_z_q75": stats["q75"],
                "pred_metric_center_z_valid_ratio": stats["valid_ratio"],
                "pred_metric_center_z_mask_count": stats["mask_count"],
                "pred_metric_center_z_finite_count": stats["finite_count"],
                "pred_metric_center_z_valid_count": stats["valid_count"],
                "pred_metric_center_z_depth_min": stats["depth_min"],
                "pred_metric_center_z_depth_max": stats["depth_max"],
                "pred_metric_center_z_valid": stats["valid"],
                "pred_metric_center_z_enabled": enabled,
                "pred_metric_center_z_route": route,
                "pred_metric_center_z_delta_log": applied_delta,
                "translation_prediction_mode": "da2_metric_center_z",
            }
        )
        return outputs

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = super().forward(inputs)
        return self.calibrate_pose_outputs(inputs, outputs)
