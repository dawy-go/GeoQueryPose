from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


CLASS_NAMES = ("bottle", "bowl", "camera", "can", "laptop", "mug")


def _category_indices(
    real_data: dict[str, Any], batch_size: int, device: torch.device
) -> torch.Tensor | None:
    category = real_data.get("category_label")
    if category is None:
        return None
    return category.to(device=device).reshape(batch_size, -1)[:, 0].long()


def _sample_weights(
    cfg: Any,
    real_data: dict[str, Any],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    config_key: str = "loss.e2e_class_weights",
    fallback_key: str | None = None,
) -> torch.Tensor:
    configured = OmegaConf.select(cfg, config_key, default=None)
    selected_key = config_key
    if configured is None and fallback_key is not None:
        configured = OmegaConf.select(cfg, fallback_key, default=None)
        selected_key = fallback_key
    category = _category_indices(real_data, batch_size, device)
    if configured is None or category is None:
        return torch.ones(batch_size, device=device, dtype=dtype)
    class_weights = torch.as_tensor(configured, device=device, dtype=dtype)
    if class_weights.ndim != 1 or class_weights.numel() == 0:
        raise ValueError(f"{selected_key} must be a non-empty list")
    if torch.any((category < 0) | (category >= class_weights.numel())):
        raise ValueError(f"category_label lies outside {selected_key}")
    return class_weights[category].clamp_min(0.0)


def _symmetry_aware_rotation_error_rad(
    prediction: torch.Tensor,
    target: torch.Tensor,
    category: torch.Tensor | None,
    handle_visibility: torch.Tensor | None = None,
    symmetric_class_ids: tuple[int, ...] = (0, 1, 3),
) -> torch.Tensor:
    """Per-instance rotation error used by the paper's 10-degree metric."""
    original_dtype = prediction.dtype
    work_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    prediction = prediction.to(dtype=work_dtype)
    target = target.to(device=prediction.device, dtype=work_dtype)
    batch_size = prediction.shape[0]
    if category is None:
        category = torch.full(
            (batch_size,), -1, device=prediction.device, dtype=torch.long
        )
    else:
        category = category.to(device=prediction.device).reshape(-1).long()

    symmetric = torch.zeros(batch_size, device=prediction.device, dtype=torch.bool)
    for class_id in symmetric_class_ids:
        symmetric |= category == class_id
    if handle_visibility is not None:
        handle_visibility = handle_visibility.to(device=prediction.device).reshape(-1)
        symmetric |= (category == 5) & (handle_visibility <= 0.5)

    errors = prediction.new_empty(batch_size)
    asymmetric = ~symmetric
    if torch.any(asymmetric):
        relative = torch.matmul(
            prediction[asymmetric].transpose(1, 2), target[asymmetric]
        )
        skew = torch.stack(
            [
                relative[:, 2, 1] - relative[:, 1, 2],
                relative[:, 0, 2] - relative[:, 2, 0],
                relative[:, 1, 0] - relative[:, 0, 1],
            ],
            dim=1,
        )
        sin_theta = 0.5 * torch.linalg.vector_norm(skew, dim=1)
        cos_theta = 0.5 * (
            relative[:, 0, 0]
            + relative[:, 1, 1]
            + relative[:, 2, 2]
            - 1.0
        )
        errors[asymmetric] = torch.atan2(
            sin_theta, cos_theta.clamp(min=-1.0, max=1.0)
        )

    if torch.any(symmetric):
        pred_axis = F.normalize(prediction[symmetric, :, 1], dim=1, eps=1.0e-6)
        target_axis = F.normalize(target[symmetric, :, 1], dim=1, eps=1.0e-6)
        sin_theta = torch.linalg.vector_norm(
            torch.linalg.cross(pred_axis, target_axis, dim=1), dim=1
        )
        cos_theta = (pred_axis * target_axis).sum(dim=1)
        errors[symmetric] = torch.atan2(
            sin_theta, cos_theta.clamp(min=-1.0, max=1.0)
        )

    return errors.to(dtype=original_dtype)


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    valid = valid.bool()
    if not torch.any(valid):
        return zero
    selected_weights = weights[valid]
    return (values[valid] * selected_weights).sum() / selected_weights.sum().clamp_min(
        1.0e-6
    )


def _smooth_l1_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    return loss.reshape(loss.shape[0], -1).mean(dim=1)


def compute_e2e_pose_aux_loss(
    cfg: Any,
    end_points: dict[str, Any],
    real_data: dict[str, Any],
):
    """Dense depth and projective-centre supervision for DA2E2EPose."""
    zero = end_points["pred_size"].new_zeros(())
    metrics = {
        "loss/e2e_depth_log": 0.0,
        "loss/e2e_surface_depth_log": 0.0,
        "loss/e2e_center_uv": 0.0,
        "loss/e2e_translation_xy": 0.0,
        "loss/e2e_translation_z": 0.0,
        "loss/e2e_translation_log_z": 0.0,
        "loss/e2e_size_log": 0.0,
        "loss/e2e_size_l1": 0.0,
        "loss/e2e_joint_threshold": 0.0,
        "loss/e2e_depth_valid_ratio": 0.0,
        "loss/e2e_center_valid_ratio": 0.0,
        "e2e/depth_mae_cm": 0.0,
        "e2e/surface_depth_mae_cm": 0.0,
        "e2e/translation_z_mae_cm": 0.0,
        "e2e/translation_z_signed_cm": 0.0,
        "e2e/size_relative_error_pct": 0.0,
        "e2e/joint_focus_ratio": 0.0,
        "e2e/joint_pass_ratio": 0.0,
        "e2e/joint_rotation_error_deg": 0.0,
        "e2e/joint_translation_error_cm": 0.0,
        "metric_center_z/enabled_ratio": 0.0,
        "metric_center_z/route_ratio": 0.0,
        "metric_center_z/valid_ratio": 0.0,
        "metric_center_z/valid_pixel_ratio": 0.0,
        "metric_center_z/finite_pixel_ratio": 0.0,
        "metric_center_z/mask_pixels": 0.0,
        "metric_center_z/depth_min_m": 0.0,
        "metric_center_z/depth_max_m": 0.0,
        "metric_center_z/mask_fallback_ratio": 0.0,
        "metric_center_z/abs_delta_log": 0.0,
    }
    for class_name in CLASS_NAMES:
        metrics[f"e2e/{class_name}_x_mae_cm"] = 0.0
        metrics[f"e2e/{class_name}_y_mae_cm"] = 0.0
        metrics[f"e2e/{class_name}_z_mae_cm"] = 0.0
        metrics[f"e2e/{class_name}_z_signed_cm"] = 0.0
        metrics[f"e2e/{class_name}_size_relative_error_pct"] = 0.0
    if end_points.get("pose_prediction_mode") != "e2e_direct_pose":
        return zero, metrics

    weight_depth = float(
        OmegaConf.select(cfg, "loss.weight_e2e_depth", default=0.0)
    )
    weight_center = float(
        OmegaConf.select(cfg, "loss.weight_e2e_center_uv", default=0.0)
    )
    weight_surface_z = float(
        OmegaConf.select(cfg, "loss.weight_e2e_surface_depth", default=0.0)
    )
    weight_translation_xy = float(
        OmegaConf.select(cfg, "loss.weight_e2e_translation_xy", default=0.0)
    )
    weight_translation_z = float(
        OmegaConf.select(cfg, "loss.weight_e2e_translation_z", default=0.0)
    )
    weight_translation_log_z = float(
        OmegaConf.select(cfg, "loss.weight_e2e_translation_log_z", default=0.0)
    )
    weight_size_log = float(
        OmegaConf.select(cfg, "loss.weight_e2e_size_log", default=0.0)
    )
    weight_size_l1 = float(
        OmegaConf.select(cfg, "loss.weight_e2e_size_l1", default=0.0)
    )
    weight_joint = float(
        OmegaConf.select(cfg, "loss.weight_e2e_joint_threshold", default=0.0)
    )
    if max(
        weight_depth,
        weight_surface_z,
        weight_center,
        weight_translation_xy,
        weight_translation_z,
        weight_translation_log_z,
        weight_size_log,
        weight_size_l1,
        weight_joint,
    ) <= 0.0:
        return zero, metrics

    batch_size = end_points["pred_size"].shape[0]
    sample_weights = _sample_weights(
        cfg,
        real_data,
        batch_size,
        zero.device,
        zero.dtype,
    )
    translation_weights = _sample_weights(
        cfg,
        real_data,
        batch_size,
        zero.device,
        zero.dtype,
        config_key="loss.e2e_translation_class_weights",
        fallback_key="loss.e2e_class_weights",
    )
    size_weights = _sample_weights(
        cfg,
        real_data,
        batch_size,
        zero.device,
        zero.dtype,
        config_key="loss.e2e_size_class_weights",
        fallback_key="loss.e2e_class_weights",
    )
    joint_weights = _sample_weights(
        cfg,
        real_data,
        batch_size,
        zero.device,
        zero.dtype,
        config_key="loss.e2e_joint_class_weights",
        fallback_key="loss.e2e_translation_class_weights",
    )
    category = _category_indices(real_data, batch_size, zero.device)

    loss_depth = zero
    loss_surface_z = zero
    pred_depth = end_points.get("pred_e2e_depth")
    target_depth = real_data.get("metric_depth")
    if (
        (weight_depth > 0.0 or weight_surface_z > 0.0)
        and pred_depth is not None
        and target_depth is not None
    ):
        target_depth = target_depth.to(
            device=pred_depth.device,
            dtype=pred_depth.dtype,
        )
        if target_depth.ndim == 4 and target_depth.shape[1] == 1:
            target_depth = target_depth[:, 0]
        if target_depth.shape[-2:] != pred_depth.shape[-2:]:
            target_depth = F.interpolate(
                target_depth.unsqueeze(1),
                size=pred_depth.shape[-2:],
                mode="nearest",
            ).squeeze(1)
        valid_depth = (
            torch.isfinite(pred_depth)
            & torch.isfinite(target_depth)
            & (pred_depth > 1.0e-6)
            & (target_depth > 1.0e-6)
        )
        if torch.any(valid_depth):
            beta = float(
                OmegaConf.select(cfg, "loss.e2e_depth_log_beta", default=0.02)
            )
            safe_pred_depth = torch.where(
                valid_depth, pred_depth, torch.ones_like(pred_depth)
            )
            safe_target_depth = torch.where(
                valid_depth, target_depth, torch.ones_like(target_depth)
            )
            pixel_loss = F.smooth_l1_loss(
                torch.log(safe_pred_depth.clamp_min(1.0e-6)),
                torch.log(safe_target_depth.clamp_min(1.0e-6)),
                beta=beta,
                reduction="none",
            )
            valid_count = valid_depth.flatten(1).sum(dim=1)
            per_sample_loss = (
                torch.where(valid_depth, pixel_loss, torch.zeros_like(pixel_loss))
                .flatten(1)
                .sum(dim=1)
                / valid_count.clamp_min(1)
            )
            valid_sample = valid_count > 0
            loss_depth = _weighted_mean(
                per_sample_loss, sample_weights, valid_sample, zero
            )
            metrics["loss/e2e_depth_log"] = float(loss_depth)
            metrics["loss/e2e_depth_valid_ratio"] = float(valid_depth.float().mean())
            pixel_mae = torch.where(
                valid_depth,
                (safe_pred_depth - safe_target_depth).abs(),
                torch.zeros_like(pred_depth),
            ).flatten(1).sum(dim=1) / valid_count.clamp_min(1)
            metrics["e2e/depth_mae_cm"] = float(
                _weighted_mean(pixel_mae, sample_weights, valid_sample, zero) * 100.0
            )

            pred_surface_z = end_points.get("pred_e2e_surface_depth_z")
            if weight_surface_z > 0.0 and pred_surface_z is not None:
                pred_surface_z = pred_surface_z.to(dtype=zero.dtype).reshape(-1)
                target_surface_z = torch.zeros_like(pred_surface_z)
                for index in range(batch_size):
                    if valid_sample[index]:
                        target_surface_z[index] = target_depth[index][
                            valid_depth[index]
                        ].median()
                valid_surface = (
                    valid_sample
                    & torch.isfinite(pred_surface_z)
                    & (pred_surface_z > 1.0e-6)
                )
                surface_loss = F.smooth_l1_loss(
                    torch.log(pred_surface_z.clamp_min(1.0e-6)),
                    torch.log(target_surface_z.clamp_min(1.0e-6)),
                    beta=beta,
                    reduction="none",
                )
                loss_surface_z = _weighted_mean(
                    surface_loss, sample_weights, valid_surface, zero
                )
                metrics["loss/e2e_surface_depth_log"] = float(loss_surface_z)
                surface_mae = (pred_surface_z - target_surface_z).abs()
                metrics["e2e/surface_depth_mae_cm"] = float(
                    _weighted_mean(
                        surface_mae, sample_weights, valid_surface, zero
                    )
                    * 100.0
                )

    loss_center = zero
    pred_center = end_points.get("pred_e2e_center_uv")
    gt_translation = real_data.get("translation_label")
    cam_k = real_data.get("cam_k")
    bbox = real_data.get("bbox")
    if (
        weight_center > 0.0
        and pred_center is not None
        and gt_translation is not None
        and cam_k is not None
        and bbox is not None
    ):
        gt_translation = gt_translation.to(
            device=pred_center.device,
            dtype=pred_center.dtype,
        ).reshape(-1, 3)
        cam_k = cam_k.to(device=pred_center.device, dtype=pred_center.dtype)
        if cam_k.ndim == 1:
            cam_k = cam_k.unsqueeze(0).expand(pred_center.shape[0], -1)
        elif cam_k.ndim == 2 and cam_k.shape == (3, 3):
            cam_k = cam_k.unsqueeze(0).expand(pred_center.shape[0], -1, -1)
        if cam_k.shape[-2:] == (3, 3):
            fx = cam_k[:, 0, 0]
            fy = cam_k[:, 1, 1]
            cx = cam_k[:, 0, 2]
            cy = cam_k[:, 1, 2]
        else:
            cam_k = cam_k.reshape(pred_center.shape[0], 4)
            fx, fy, cx, cy = cam_k.unbind(dim=1)
        gt_z = gt_translation[:, 2]
        gt_center = torch.stack(
            [
                fx * gt_translation[:, 0] / gt_z.clamp_min(1.0e-6) + cx,
                fy * gt_translation[:, 1] / gt_z.clamp_min(1.0e-6) + cy,
            ],
            dim=1,
        )
        bbox = bbox.to(device=pred_center.device, dtype=pred_center.dtype).reshape(
            -1, 4
        )
        y1, x1, y2, x2 = bbox.unbind(dim=1)
        bbox_size = torch.stack(
            [(x2 - x1).clamp_min(1.0), (y2 - y1).clamp_min(1.0)],
            dim=1,
        )
        valid_center = (
            torch.isfinite(pred_center).all(dim=1)
            & torch.isfinite(gt_center).all(dim=1)
            & torch.isfinite(gt_translation).all(dim=1)
            & (gt_z > 1.0e-6)
        )
        if torch.any(valid_center):
            beta = float(
                OmegaConf.select(cfg, "loss.e2e_center_uv_beta", default=0.02)
            )
            normalized_error = (
                pred_center[valid_center] - gt_center[valid_center]
            ) / bbox_size[valid_center]
            center_loss = F.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                beta=beta,
                reduction="none",
            ).mean(dim=1)
            loss_center = _weighted_mean(
                center_loss,
                translation_weights[valid_center],
                torch.ones_like(center_loss, dtype=torch.bool),
                zero,
            )
            metrics["loss/e2e_center_uv"] = float(loss_center)
            metrics["loss/e2e_center_valid_ratio"] = float(
                valid_center.float().mean()
            )

    loss_translation_xy = zero
    loss_translation_z = zero
    loss_translation_log_z = zero
    pred_translation = end_points.get("pred_translation")
    gt_translation = real_data.get("translation_label")
    if pred_translation is not None and gt_translation is not None:
        gt_translation = gt_translation.to(
            device=pred_translation.device, dtype=pred_translation.dtype
        ).reshape(-1, 3)
        valid_translation = (
            torch.isfinite(pred_translation).all(dim=1)
            & torch.isfinite(gt_translation).all(dim=1)
            & (pred_translation[:, 2] > 1.0e-6)
            & (gt_translation[:, 2] > 1.0e-6)
        )
        beta = float(
            OmegaConf.select(cfg, "loss.e2e_translation_beta", default=0.02)
        )
        safe_pred_translation = torch.where(
            valid_translation[:, None], pred_translation, gt_translation
        )
        xy_values = _smooth_l1_per_sample(
            safe_pred_translation[:, :2], gt_translation[:, :2], beta
        )
        z_values = F.smooth_l1_loss(
            safe_pred_translation[:, 2],
            gt_translation[:, 2],
            beta=beta,
            reduction="none",
        )
        log_z_values = F.smooth_l1_loss(
            torch.log(safe_pred_translation[:, 2].clamp_min(1.0e-6)),
            torch.log(gt_translation[:, 2].clamp_min(1.0e-6)),
            beta=beta,
            reduction="none",
        )
        loss_translation_xy = _weighted_mean(
            xy_values, translation_weights, valid_translation, zero
        )
        loss_translation_z = _weighted_mean(
            z_values, translation_weights, valid_translation, zero
        )
        loss_translation_log_z = _weighted_mean(
            log_z_values, translation_weights, valid_translation, zero
        )
        metrics["loss/e2e_translation_xy"] = float(loss_translation_xy)
        metrics["loss/e2e_translation_z"] = float(loss_translation_z)
        metrics["loss/e2e_translation_log_z"] = float(loss_translation_log_z)
        z_error = pred_translation[:, 2] - gt_translation[:, 2]
        metrics["e2e/translation_z_mae_cm"] = float(
            _weighted_mean(
                z_error.abs(), translation_weights, valid_translation, zero
            )
            * 100.0
        )
        metrics["e2e/translation_z_signed_cm"] = float(
            _weighted_mean(z_error, translation_weights, valid_translation, zero)
            * 100.0
        )
        if category is not None:
            xyz_error = (pred_translation - gt_translation).abs()
            for class_id, class_name in enumerate(CLASS_NAMES):
                class_valid = valid_translation & (category == class_id)
                if not torch.any(class_valid):
                    continue
                metrics[f"e2e/{class_name}_x_mae_cm"] = float(
                    xyz_error[class_valid, 0].mean() * 100.0
                )
                metrics[f"e2e/{class_name}_y_mae_cm"] = float(
                    xyz_error[class_valid, 1].mean() * 100.0
                )
                metrics[f"e2e/{class_name}_z_mae_cm"] = float(
                    xyz_error[class_valid, 2].mean() * 100.0
                )
                metrics[f"e2e/{class_name}_z_signed_cm"] = float(
                    z_error[class_valid].mean() * 100.0
                )

        base_translation = end_points.get("pred_translation_base")
        if base_translation is not None:
            base_translation = base_translation.to(
                device=pred_translation.device,
                dtype=pred_translation.dtype,
            ).reshape(-1, 3)
            valid_base = (
                valid_translation
                & torch.isfinite(base_translation).all(dim=1)
                & (base_translation[:, 2] > 1.0e-6)
            )
            base_z_error = (base_translation[:, 2] - gt_translation[:, 2]).abs()
            final_z_error = (pred_translation[:, 2] - gt_translation[:, 2]).abs()
            metrics["metric_center_z/base_z_mae_cm"] = float(
                _weighted_mean(
                    base_z_error,
                    translation_weights,
                    valid_base,
                    zero,
                )
                * 100.0
            )
            metrics["metric_center_z/z_improvement_cm"] = float(
                _weighted_mean(
                    base_z_error - final_z_error,
                    translation_weights,
                    valid_base,
                    zero,
                )
                * 100.0
            )
            if category is not None:
                for class_id, class_name in enumerate(CLASS_NAMES):
                    class_valid = valid_base & (category == class_id)
                    if torch.any(class_valid):
                        metrics[
                            f"metric_center_z/{class_name}_z_improvement_cm"
                        ] = float(
                            (
                                base_z_error[class_valid]
                                - final_z_error[class_valid]
                            ).mean()
                            * 100.0
                        )

        route = end_points.get("pred_metric_center_z_route")
        enabled = end_points.get("pred_metric_center_z_enabled")
        metric_valid = end_points.get("pred_metric_center_z_valid")
        valid_pixel_ratio = end_points.get("pred_metric_center_z_valid_ratio")
        mask_count = end_points.get("pred_metric_center_z_mask_count")
        finite_count = end_points.get("pred_metric_center_z_finite_count")
        depth_min = end_points.get("pred_metric_center_z_depth_min")
        depth_max = end_points.get("pred_metric_center_z_depth_max")
        delta_log = end_points.get("pred_metric_center_z_delta_log")
        if enabled is not None:
            metrics["metric_center_z/enabled_ratio"] = float(enabled.float().mean())
        if route is not None:
            metrics["metric_center_z/route_ratio"] = float(route.float().mean())
        if metric_valid is not None:
            metrics["metric_center_z/valid_ratio"] = float(
                metric_valid.float().mean()
            )
        if valid_pixel_ratio is not None:
            metrics["metric_center_z/valid_pixel_ratio"] = float(
                valid_pixel_ratio.float().mean()
            )
        if mask_count is not None:
            mask_count = mask_count.float()
            metrics["metric_center_z/mask_pixels"] = float(mask_count.mean())
            if finite_count is not None:
                metrics["metric_center_z/finite_pixel_ratio"] = float(
                    (finite_count.float() / mask_count.clamp_min(1.0)).mean()
                )
        if depth_min is not None:
            finite_min = depth_min[torch.isfinite(depth_min)]
            if finite_min.numel() > 0:
                metrics["metric_center_z/depth_min_m"] = float(finite_min.min())
        if depth_max is not None:
            finite_max = depth_max[torch.isfinite(depth_max)]
            if finite_max.numel() > 0:
                metrics["metric_center_z/depth_max_m"] = float(finite_max.max())
        fallback_used = real_data.get("scene_mask_fallback_used")
        if fallback_used is not None:
            metrics["metric_center_z/mask_fallback_ratio"] = float(
                fallback_used.to(device=zero.device).float().mean()
            )
        if delta_log is not None:
            metrics["metric_center_z/abs_delta_log"] = float(
                delta_log.detach().abs().mean()
            )

    loss_size_log = zero
    loss_size_l1 = zero
    pred_size = end_points.get("pred_metric_size", end_points.get("pred_size"))
    gt_size = real_data.get("size_label")
    if pred_size is not None and gt_size is not None:
        gt_size = gt_size.to(device=pred_size.device, dtype=pred_size.dtype).reshape(
            -1, 3
        )
        valid_size = (
            torch.isfinite(pred_size).all(dim=1)
            & torch.isfinite(gt_size).all(dim=1)
            & (pred_size > 0.0).all(dim=1)
            & (gt_size > 0.0).all(dim=1)
        )
        beta = float(OmegaConf.select(cfg, "loss.e2e_size_beta", default=0.02))
        safe_pred_size = torch.where(valid_size[:, None], pred_size, gt_size)
        log_ratio = torch.log(safe_pred_size.clamp_min(1.0e-6)) - torch.log(
            gt_size.clamp_min(1.0e-6)
        )
        size_log_values = F.smooth_l1_loss(
            log_ratio,
            torch.zeros_like(log_ratio),
            beta=beta,
            reduction="none",
        ).mean(dim=1)
        size_l1_values = _smooth_l1_per_sample(safe_pred_size, gt_size, beta)
        loss_size_log = _weighted_mean(
            size_log_values, size_weights, valid_size, zero
        )
        loss_size_l1 = _weighted_mean(
            size_l1_values, size_weights, valid_size, zero
        )
        metrics["loss/e2e_size_log"] = float(loss_size_log)
        metrics["loss/e2e_size_l1"] = float(loss_size_l1)
        relative_size = ((pred_size - gt_size).abs() / gt_size.clamp_min(1.0e-6)).mean(
            dim=1
        )
        metrics["e2e/size_relative_error_pct"] = float(
            _weighted_mean(relative_size, size_weights, valid_size, zero) * 100.0
        )
        if category is not None:
            for class_id, class_name in enumerate(CLASS_NAMES):
                class_valid = valid_size & (category == class_id)
                if torch.any(class_valid):
                    metrics[f"e2e/{class_name}_size_relative_error_pct"] = float(
                        relative_size[class_valid].mean() * 100.0
                    )

    loss_joint = zero
    pred_rotation = end_points.get("pred_rotation")
    gt_rotation = real_data.get("rotation_label")
    pred_translation = end_points.get("pred_translation")
    gt_translation = real_data.get("translation_label")
    if (
        weight_joint > 0.0
        and pred_rotation is not None
        and gt_rotation is not None
        and pred_translation is not None
        and gt_translation is not None
    ):
        gt_rotation = gt_rotation.to(
            device=pred_rotation.device, dtype=pred_rotation.dtype
        ).reshape(-1, 3, 3)
        gt_translation = gt_translation.to(
            device=pred_translation.device, dtype=pred_translation.dtype
        ).reshape(-1, 3)
        valid_joint = (
            torch.isfinite(pred_rotation.reshape(batch_size, -1)).all(dim=1)
            & torch.isfinite(gt_rotation.reshape(batch_size, -1)).all(dim=1)
            & torch.isfinite(pred_translation).all(dim=1)
            & torch.isfinite(gt_translation).all(dim=1)
        )
        rotation_error = _symmetry_aware_rotation_error_rad(
            pred_rotation,
            gt_rotation,
            category,
            real_data.get("gt_handle_visibility"),
        )
        translation_error = torch.linalg.vector_norm(
            pred_translation - gt_translation, dim=1
        )

        rotation_threshold_deg = float(
            OmegaConf.select(
                cfg, "loss.e2e_joint_rotation_threshold_deg", default=10.0
            )
        )
        translation_threshold_m = float(
            OmegaConf.select(
                cfg, "loss.e2e_joint_translation_threshold_m", default=0.10
            )
        )
        rotation_max_deg = float(
            OmegaConf.select(
                cfg, "loss.e2e_joint_rotation_max_deg", default=25.0
            )
        )
        translation_max_m = float(
            OmegaConf.select(
                cfg, "loss.e2e_joint_translation_max_m", default=0.15
            )
        )
        min_ratio = float(
            OmegaConf.select(cfg, "loss.e2e_joint_min_ratio", default=0.8)
        )
        temperature = float(
            OmegaConf.select(cfg, "loss.e2e_joint_temperature", default=0.2)
        )
        if min(rotation_threshold_deg, translation_threshold_m, temperature) <= 0.0:
            raise ValueError("E2E joint thresholds and temperature must be positive")
        if rotation_max_deg < rotation_threshold_deg:
            raise ValueError("loss.e2e_joint_rotation_max_deg must be at least the threshold")
        if translation_max_m < translation_threshold_m:
            raise ValueError("loss.e2e_joint_translation_max_m must be at least the threshold")
        if min_ratio < 0.0:
            raise ValueError("loss.e2e_joint_min_ratio must be non-negative")

        radians_per_degree = torch.pi / 180.0
        rotation_ratio = rotation_error / (
            rotation_threshold_deg * radians_per_degree
        )
        translation_ratio = translation_error / translation_threshold_m
        detached_rotation = rotation_error.detach()
        detached_translation = translation_error.detach()
        focus = (
            valid_joint
            & (detached_rotation <= rotation_max_deg * radians_per_degree)
            & (detached_translation <= translation_max_m)
            & (
                (rotation_ratio.detach() >= min_ratio)
                | (translation_ratio.detach() >= min_ratio)
            )
        )
        joint_values = temperature * torch.logsumexp(
            torch.stack([rotation_ratio, translation_ratio], dim=1) / temperature,
            dim=1,
        )
        loss_joint = _weighted_mean(joint_values, joint_weights, focus, zero)
        metrics["loss/e2e_joint_threshold"] = float(loss_joint)
        if torch.any(valid_joint):
            valid_count = valid_joint.float().sum().clamp_min(1.0)
            joint_pass = (
                valid_joint
                & (detached_rotation <= rotation_threshold_deg * radians_per_degree)
                & (detached_translation <= translation_threshold_m)
            )
            metrics["e2e/joint_focus_ratio"] = float(
                focus.float().sum() / valid_count
            )
            metrics["e2e/joint_pass_ratio"] = float(
                joint_pass.float().sum() / valid_count
            )
            metrics["e2e/joint_rotation_error_deg"] = float(
                _weighted_mean(
                    rotation_error / radians_per_degree,
                    joint_weights,
                    valid_joint,
                    zero,
                )
            )
            metrics["e2e/joint_translation_error_cm"] = float(
                _weighted_mean(
                    translation_error * 100.0,
                    joint_weights,
                    valid_joint,
                    zero,
                )
            )

    loss = (
        weight_depth * loss_depth
        + weight_surface_z * loss_surface_z
        + weight_center * loss_center
        + weight_translation_xy * loss_translation_xy
        + weight_translation_z * loss_translation_z
        + weight_translation_log_z * loss_translation_log_z
        + weight_size_log * loss_size_log
        + weight_size_l1 * loss_size_l1
        + weight_joint * loss_joint
    )
    return loss, metrics
