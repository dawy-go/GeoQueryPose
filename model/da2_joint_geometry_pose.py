from __future__ import annotations

import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.da2_metric_center_z_pose import DA2MetricCenterZPose, _select
from modules.joint_geometry_pose import (
    GeometryResidualHead,
    JointCategoryHead,
    JointPointNOCSHead,
    solve_nocs_geometry,
)


class DA2JointGeometryPose(DA2MetricCenterZPose):
    """Joint category routing, dense NOCS geometry, and guarded pose refinement."""

    def __init__(
        self,
        cfg: Any,
        visual_encoder: Optional[nn.Module] = None,
        metric_depth_predictor: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            cfg,
            visual_encoder=visual_encoder,
            metric_depth_predictor=metric_depth_predictor,
        )
        hidden_dim = int(_select(cfg, "joint_geometry.hidden_dim", 256))
        self.point_count = int(_select(cfg, "joint_geometry.point_count", 1024))
        self.min_correspondence_points = int(
            _select(cfg, "joint_geometry.min_correspondence_points", 64)
        )
        if self.point_count <= 0:
            raise ValueError("joint_geometry.point_count must be positive")
        if self.min_correspondence_points <= 0:
            raise ValueError(
                "joint_geometry.min_correspondence_points must be positive"
            )
        hypothesis_ids = tuple(
            int(value)
            for value in _select(
                cfg, "joint_geometry.hypothesis_class_ids", [0, 3]
            )
        )
        if len(hypothesis_ids) != 2 or any(
            value < 0 or value >= self.num_classes for value in hypothesis_ids
        ):
            raise ValueError(
                "joint_geometry.hypothesis_class_ids must contain two valid classes"
            )
        self.hypothesis_class_ids = hypothesis_ids
        self.quality_score_weight = float(
            _select(cfg, "joint_geometry.quality_score_weight", 0.0)
        )
        self.geometry_residual_score_weight = float(
            _select(cfg, "joint_geometry.geometry_residual_score_weight", 0.0)
        )
        self.rotation_gate_scale = float(
            _select(cfg, "joint_geometry.rotation_gate_scale", 1.0)
        )
        if not 0.0 <= self.rotation_gate_scale <= 1.0:
            raise ValueError("joint_geometry.rotation_gate_scale must lie in [0, 1]")
        self.rotation_source = str(
            _select(cfg, "joint_geometry.rotation_source", "selected")
        ).lower()
        if self.rotation_source not in {"selected", "detector"}:
            raise ValueError(
                "joint_geometry.rotation_source must be selected or detector"
            )

        self.joint_category_head = JointCategoryHead(
            feature_dim=self.token_dim,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            prior_logit=float(_select(cfg, "joint_geometry.detector_prior_logit", 4.0)),
        )
        self.joint_nocs_head = JointPointNOCSHead(
            global_dim=self.token_dim,
            hidden_dim=hidden_dim,
            global_proj_dim=hidden_dim,
        )
        self.joint_geometry_head = GeometryResidualHead(
            query_dim=self.token_dim,
            num_classes=self.num_classes,
            category_dim=int(_select(cfg, "joint_geometry.category_dim", 16)),
            hidden_dim=hidden_dim,
            dropout=float(_select(cfg, "joint_geometry.dropout", 0.1)),
            max_translation_ratio=float(
                _select(cfg, "joint_geometry.max_translation_ratio", 0.5)
            ),
            max_log_size_residual=float(
                _select(cfg, "joint_geometry.max_log_size_residual", 0.35)
            ),
            initial_gate_logit=float(
                _select(cfg, "joint_geometry.initial_gate_logit", -8.0)
            ),
            rotation_gate_scale=self.rotation_gate_scale,
        )
        self.joint_trainable_scope = str(
            _select(cfg, "joint_geometry.trainable_scope", "joint_heads")
        ).lower()
        valid_scopes = {"joint_heads", "joint_and_metric", "pose_and_joint"}
        if self.joint_trainable_scope not in valid_scopes:
            raise ValueError(
                "joint_geometry.trainable_scope must be one of "
                f"{sorted(valid_scopes)}"
            )
        self._freeze_for_joint_geometry()

    def _freeze_for_joint_geometry(self) -> None:
        self.requires_grad_(False)
        for module in (
            self.joint_category_head,
            self.joint_nocs_head,
            self.joint_geometry_head,
        ):
            module.requires_grad_(True)
        if self.joint_trainable_scope in {"joint_and_metric", "pose_and_joint"}:
            self.metric_center_z_head.requires_grad_(True)
        if self.joint_trainable_scope == "pose_and_joint":
            for name, parameter in self.named_parameters():
                if name in self._base_trainable_parameter_names:
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in (
            self.joint_category_head,
            self.joint_nocs_head,
            self.joint_geometry_head,
        ):
            module.train(mode)
        if self.joint_trainable_scope == "joint_heads":
            self.metric_center_z_head.eval()
        return self

    @staticmethod
    def _batch_vector(
        value: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        default: float,
    ) -> torch.Tensor:
        if value is None:
            return torch.full(
                (batch_size,), default, device=device, dtype=dtype
            )
        result = value.to(device=device, dtype=dtype).reshape(-1)
        if result.numel() == 1 and batch_size > 1:
            result = result.expand(batch_size)
        if result.numel() != batch_size:
            raise ValueError("per-object value count does not match batch size")
        return result

    def _detector_category(
        self, inputs: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        category = inputs.get("detector_category_label", inputs.get("category_label"))
        if category is None:
            raise KeyError(
                "DA2JointGeometryPose requires detector_category_label or category_label"
            )
        category = category.to(device=device).reshape(-1).long()
        if category.numel() != batch_size:
            raise ValueError("detector category count does not match batch size")
        if torch.any((category < 0) | (category >= self.num_classes)):
            raise ValueError("detector category lies outside the configured class range")
        return category

    @staticmethod
    def _select_branch_tensor(
        original: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        selected_category: torch.Tensor,
        first_class: int,
        second_class: int,
    ) -> torch.Tensor:
        shape = (selected_category.shape[0],) + (1,) * (original.ndim - 1)
        first_mask = (selected_category == first_class).reshape(shape)
        second_mask = (selected_category == second_class).reshape(shape)
        return torch.where(
            second_mask,
            second,
            torch.where(first_mask, first, original),
        )

    def _merge_branches(
        self,
        original: dict[str, Any],
        first: dict[str, Any],
        second: dict[str, Any],
        selected_category: torch.Tensor,
    ) -> dict[str, Any]:
        first_class, second_class = self.hypothesis_class_ids
        merged: dict[str, Any] = {}
        for key, original_value in original.items():
            first_value = first.get(key)
            second_value = second.get(key)
            if (
                isinstance(original_value, torch.Tensor)
                and isinstance(first_value, torch.Tensor)
                and isinstance(second_value, torch.Tensor)
                and original_value.ndim > 0
                and original_value.shape[0] == selected_category.shape[0]
                and first_value.shape == original_value.shape
                and second_value.shape == original_value.shape
            ):
                merged[key] = self._select_branch_tensor(
                    original_value,
                    first_value,
                    second_value,
                    selected_category,
                    first_class,
                    second_class,
                )
            else:
                merged[key] = original_value
        return merged

    def _sample_metric_points(
        self,
        inputs: dict[str, torch.Tensor],
        scene_rgb: torch.Tensor,
        object_masks: torch.Tensor,
        metric_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        object_count = object_masks.shape[0]
        if object_masks.shape[-2:] != metric_depth.shape[-2:]:
            object_masks = F.interpolate(
                object_masks.float().unsqueeze(1),
                size=metric_depth.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()
        if scene_rgb.shape[-2:] != metric_depth.shape[-2:]:
            scene_rgb = F.interpolate(
                scene_rgb.float(),
                size=metric_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        map_indices = (
            torch.zeros(object_count, device=metric_depth.device, dtype=torch.long)
            if metric_depth.shape[0] == 1
            else torch.arange(object_count, device=metric_depth.device)
        )
        rgb_indices = (
            torch.zeros(object_count, device=scene_rgb.device, dtype=torch.long)
            if scene_rgb.shape[0] == 1
            else torch.arange(object_count, device=scene_rgb.device)
        )
        selected_depth = metric_depth[map_indices].float()
        selected_rgb = scene_rgb[rgb_indices].to(selected_depth)
        object_masks = object_masks.to(selected_depth.device)
        intrinsics = self._intrinsics(
            inputs,
            object_count,
            selected_depth.device,
            selected_depth.dtype,
        )
        fx, fy, cx, cy = intrinsics

        camera_points = []
        point_features = []
        sample_valid = []
        source_valid_ratio = []
        for index in range(object_count):
            depth = selected_depth[index]
            valid = (
                object_masks[index]
                & torch.isfinite(depth)
                & (depth >= self.metric_depth_min)
                & (depth <= self.metric_depth_max)
            )
            coordinates = torch.nonzero(valid, as_tuple=False)
            count = int(coordinates.shape[0])
            sample_valid.append(count >= self.min_correspondence_points)
            source_valid_ratio.append(
                valid.float().sum()
                / object_masks[index].float().sum().clamp_min(1.0)
            )
            if count == 0:
                coordinates = torch.zeros(
                    (1, 2), device=depth.device, dtype=torch.long
                )
                count = 1
            positions = torch.linspace(
                0,
                count - 1,
                self.point_count,
                device=depth.device,
            ).long()
            coordinates = coordinates[positions]
            rows = coordinates[:, 0]
            cols = coordinates[:, 1]
            z = depth[rows, cols]
            points = torch.stack(
                [
                    (cols.to(z.dtype) - cx[index]) * z / fx[index],
                    (rows.to(z.dtype) - cy[index]) * z / fy[index],
                    z,
                ],
                dim=1,
            )
            local = points - points.mean(dim=0, keepdim=True)
            local = local / torch.linalg.vector_norm(local, dim=1).amax().clamp_min(
                1.0e-4
            )
            rgb_points = selected_rgb[index, :, rows, cols].transpose(0, 1)
            normals = torch.zeros_like(local)
            camera_points.append(points)
            point_features.append(torch.cat([local, rgb_points, normals], dim=1))

        return {
            "camera_points": torch.stack(camera_points),
            "point_features": torch.stack(point_features),
            "sample_valid": torch.as_tensor(
                sample_valid,
                device=selected_depth.device,
                dtype=torch.bool,
            ),
            "source_valid_ratio": torch.stack(source_valid_ratio),
        }

    def _refine_branch(
        self,
        direct: dict[str, Any],
        category: torch.Tensor,
        class_probability: torch.Tensor,
        sampled: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        pred_nocs, confidence_logits = self.joint_nocs_head(
            sampled["point_features"], direct["pred_pose_query"]
        )
        geometry = solve_nocs_geometry(
            pred_nocs,
            confidence_logits,
            direct["pred_size"],
            sampled["camera_points"],
        )
        geometry_valid = geometry["valid"] & sampled["sample_valid"]
        refined = self.joint_geometry_head(
            query=direct["pred_pose_query"],
            category=category,
            base_translation=direct["pred_translation"],
            base_rotation=direct["pred_rotation"],
            base_size=direct["pred_size"],
            geometry_translation=geometry["translation"],
            geometry_rotation=geometry["rotation"],
            geometry_residual=geometry["residual"],
            geometry_valid_ratio=sampled["source_valid_ratio"],
            geometry_valid=geometry_valid,
            class_probability=class_probability,
        )
        output = dict(direct)
        output["pred_translation_center_z"] = direct["pred_translation"]
        output["pred_size_base"] = direct["pred_size"]
        output["pred_translation"] = refined["translation"]
        output["pred_rotation_base"] = direct["pred_rotation"]
        output["pred_rotation"] = refined["rotation"]
        output["pred_size"] = refined["size"]
        output["pred_metric_size"] = refined["size"]
        output["pred_nocs"] = pred_nocs
        output["pred_nocs_confidence_logits"] = confidence_logits
        output["pred_nocs_confidence"] = geometry["confidence"]
        output["pred_nocs_rotation"] = geometry["rotation"]
        output["pred_nocs_translation"] = geometry["translation"]
        output["pred_nocs_residual"] = geometry["residual"]
        output["pred_nocs_valid_ratio"] = geometry["valid_ratio"]
        output["pred_joint_geometry_valid"] = geometry_valid
        output["pred_joint_geometry_gate"] = refined["gate"]
        output["pred_joint_translation_delta"] = refined["translation_delta"]
        output["pred_joint_log_size_delta"] = refined["log_size_delta"]
        output["pred_joint_quality_logit"] = refined["quality_logit"]
        output["translation_prediction_mode"] = "joint_geometry_xyz"
        return output

    @staticmethod
    def _candidate_score(
        branch: dict[str, Any],
        class_probability: torch.Tensor,
        quality_weight: float,
        residual_weight: float,
    ) -> torch.Tensor:
        return (
            torch.log(class_probability.clamp_min(1.0e-8))
            + quality_weight * branch["pred_joint_quality_logit"]
            - residual_weight * branch["pred_nocs_residual"]
        )

    @staticmethod
    def _add_prefixed_candidate_outputs(
        outputs: dict[str, Any], prefix: str, branch: dict[str, Any]
    ) -> None:
        keys = (
            "pred_translation",
            "pred_rotation",
            "pred_size",
            "pred_nocs",
            "pred_nocs_confidence_logits",
            "pred_nocs_rotation",
            "pred_nocs_translation",
            "pred_nocs_residual",
            "pred_joint_geometry_valid",
            "pred_joint_geometry_gate",
            "pred_joint_quality_logit",
        )
        for key in keys:
            outputs[f"{prefix}_{key}"] = branch[key]

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        start = time.time()
        encoded = self.encode_pose_features(inputs)
        batch_size = int(encoded["batch_size"])
        device = encoded["tokens"].device
        detector_category = self._detector_category(inputs, batch_size, device)
        detector_score = self._batch_vector(
            inputs.get("detector_score", inputs.get("pred_scores")),
            batch_size,
            device,
            encoded["tokens"].dtype,
            1.0,
        )
        class_logits = self.joint_category_head(
            encoded["global_feature"], detector_category, detector_score
        )
        class_probabilities = torch.softmax(class_logits, dim=1)

        scene_rgb, object_masks = self._scene_inputs(inputs, batch_size)
        metric_depth = self._metric_depth(inputs, scene_rgb)
        depth_stats = self._masked_depth_statistics(metric_depth, object_masks)
        sampled = self._sample_metric_points(
            inputs, scene_rgb, object_masks, metric_depth
        )

        first_class, second_class = self.hypothesis_class_ids
        first_category = torch.full_like(detector_category, first_class)
        second_category = torch.full_like(detector_category, second_class)
        direct_original = self.calibrate_pose_outputs(
            inputs,
            self.decode_pose_features(encoded, detector_category),
            detector_category,
            metric_depth=metric_depth,
            stats=depth_stats,
        )
        direct_first = self.calibrate_pose_outputs(
            inputs,
            self.decode_pose_features(encoded, first_category),
            first_category,
            metric_depth=metric_depth,
            stats=depth_stats,
        )
        direct_second = self.calibrate_pose_outputs(
            inputs,
            self.decode_pose_features(encoded, second_category),
            second_category,
            metric_depth=metric_depth,
            stats=depth_stats,
        )
        probability_original = class_probabilities.gather(
            1, detector_category[:, None]
        ).squeeze(1)
        probability_first = class_probabilities[:, first_class]
        probability_second = class_probabilities[:, second_class]
        branch_original = self._refine_branch(
            direct_original, detector_category, probability_original, sampled
        )
        branch_first = self._refine_branch(
            direct_first, first_category, probability_first, sampled
        )
        branch_second = self._refine_branch(
            direct_second, second_category, probability_second, sampled
        )

        first_score = self._candidate_score(
            branch_first,
            probability_first,
            self.quality_score_weight,
            self.geometry_residual_score_weight,
        )
        second_score = self._candidate_score(
            branch_second,
            probability_second,
            self.quality_score_weight,
            self.geometry_residual_score_weight,
        )
        ambiguous = (detector_category == first_class) | (
            detector_category == second_class
        )
        selected_category = detector_category.clone()
        selected_category[ambiguous] = torch.where(
            first_score[ambiguous] >= second_score[ambiguous],
            first_category[ambiguous],
            second_category[ambiguous],
        )
        outputs = self._merge_branches(
            branch_original,
            branch_first,
            branch_second,
            selected_category,
        )
        if self.rotation_source == "detector":
            # Keep class/translation/size routing from the selected branch,
            # while using the detector-conditioned frozen rotation path.
            outputs["pred_rotation"] = branch_original["pred_rotation"]
            outputs["pred_rotation_base"] = branch_original["pred_rotation_base"]
        outputs["pred_class_logits"] = class_logits
        outputs["pred_class_probabilities"] = class_probabilities
        outputs["pred_detector_category"] = detector_category
        outputs["pred_joint_category"] = selected_category
        outputs["pred_joint_class_ids"] = selected_category + 1
        outputs["pred_joint_class_score"] = class_probabilities.gather(
            1, selected_category[:, None]
        ).squeeze(1)
        outputs["pred_joint_first_candidate_score"] = first_score
        outputs["pred_joint_second_candidate_score"] = second_score
        outputs["pred_joint_camera_points"] = sampled["camera_points"]
        outputs["pred_joint_sample_valid"] = sampled["sample_valid"]
        outputs["pred_joint_source_valid_ratio"] = sampled["source_valid_ratio"]
        outputs["joint_geometry_rotation_source"] = self.rotation_source
        self._add_prefixed_candidate_outputs(outputs, "bottle", branch_first)
        self._add_prefixed_candidate_outputs(outputs, "can", branch_second)
        outputs.update(
            {
                "pose_prediction_mode": "joint_geometry_pose",
                "resnet_forward_time": time.time() - start,
                "octree_forward_time": 0.0,
                "shapenet_forward_time": 0.0,
                "diffusion_forward_time": 0.0,
                "denoiser_time": 0.0,
            }
        )
        return outputs
