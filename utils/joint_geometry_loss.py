from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


SYMMETRIC_CLASS_IDS = (0, 1, 3)


def _configured_weight(cfg: Any, key: str, default: float = 0.0) -> float:
    return float(OmegaConf.select(cfg, f"loss.{key}", default=default))


def _correct_candidate(
    end_points: dict[str, Any],
    key: str,
    category: torch.Tensor,
) -> torch.Tensor:
    value = end_points[key]
    bottle = end_points.get(f"bottle_{key}", value)
    can = end_points.get(f"can_{key}", value)
    bottle_mask = (category == 0).reshape(
        (category.shape[0],) + (1,) * (value.ndim - 1)
    )
    can_mask = (category == 3).reshape(
        (category.shape[0],) + (1,) * (value.ndim - 1)
    )
    return torch.where(can_mask, can, torch.where(bottle_mask, bottle, value))


def _canonical_targets(
    camera_points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    size: torch.Tensor,
) -> torch.Tensor:
    centered = camera_points - translation[:, None, :]
    metric_canonical = torch.bmm(centered, rotation)
    return metric_canonical / size[:, None, :].clamp_min(1.0e-4)


def _nocs_per_point_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    category: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    coordinate = F.smooth_l1_loss(
        prediction,
        target,
        beta=beta,
        reduction="none",
    ).mean(dim=2)
    symmetric = torch.zeros_like(category, dtype=torch.bool)
    for class_id in SYMMETRIC_CLASS_IDS:
        symmetric |= category == class_id
    if torch.any(symmetric):
        pred_radius = torch.linalg.vector_norm(prediction[symmetric][..., [0, 2]], dim=2)
        target_radius = torch.linalg.vector_norm(target[symmetric][..., [0, 2]], dim=2)
        radius = F.smooth_l1_loss(
            pred_radius,
            target_radius,
            beta=beta,
            reduction="none",
        )
        vertical = F.smooth_l1_loss(
            prediction[symmetric][..., 1],
            target[symmetric][..., 1],
            beta=beta,
            reduction="none",
        )
        coordinate = coordinate.clone()
        coordinate[symmetric] = 0.5 * (radius + vertical)
    return coordinate


def compute_joint_geometry_loss(
    cfg: Any,
    end_points: dict[str, Any],
    real_data: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = end_points.get("pred_translation")
    if end_points.get("pose_prediction_mode") != "joint_geometry_pose" or prediction is None:
        device = prediction.device if prediction is not None else torch.device("cpu")
        zero = torch.zeros((), device=device)
        return zero, {"loss/joint_geometry": 0.0}

    zero = prediction.new_zeros(())
    category = real_data["category_label"].to(prediction.device).reshape(-1).long()
    gt_translation = real_data["translation_label"].to(prediction).reshape(-1, 3)
    gt_rotation = real_data["rotation_label"].to(prediction).reshape(-1, 3, 3)
    gt_size = real_data["size_label"].to(prediction).reshape(-1, 3)

    class_logits = end_points["pred_class_logits"]
    loss_classification = F.cross_entropy(class_logits, category)

    pred_translation = _correct_candidate(
        end_points, "pred_translation", category
    )
    pred_rotation = _correct_candidate(end_points, "pred_rotation", category)
    pred_size = _correct_candidate(end_points, "pred_size", category)
    valid_pose = (
        torch.isfinite(pred_translation).all(dim=1)
        & torch.isfinite(pred_rotation).flatten(1).all(dim=1)
        & torch.isfinite(pred_size).all(dim=1)
        & (pred_size > 0.0).all(dim=1)
    )
    translation_beta = float(
        OmegaConf.select(cfg, "loss.joint_translation_beta", default=0.02)
    )
    size_beta = float(
        OmegaConf.select(cfg, "loss.joint_size_log_beta", default=0.02)
    )
    if torch.any(valid_pose):
        loss_translation = F.smooth_l1_loss(
            pred_translation[valid_pose],
            gt_translation[valid_pose],
            beta=translation_beta,
        )
        loss_size = F.smooth_l1_loss(
            torch.log(pred_size[valid_pose].clamp_min(1.0e-6)),
            torch.log(gt_size[valid_pose].clamp_min(1.0e-6)),
            beta=size_beta,
        )
    else:
        loss_translation = zero
        loss_size = zero

    camera_points = end_points["pred_joint_camera_points"]
    sample_valid = end_points["pred_joint_sample_valid"].bool()
    target_nocs = _canonical_targets(
        camera_points,
        gt_rotation,
        gt_translation,
        gt_size,
    )
    pred_nocs = _correct_candidate(end_points, "pred_nocs", category)
    confidence_logits = _correct_candidate(
        end_points, "pred_nocs_confidence_logits", category
    )
    point_valid = (
        torch.isfinite(target_nocs).all(dim=2)
        & (target_nocs.abs() <= 0.65).all(dim=2)
        & sample_valid[:, None]
    )
    nocs_beta = float(OmegaConf.select(cfg, "loss.joint_nocs_beta", default=0.02))
    nocs_values = _nocs_per_point_loss(
        pred_nocs,
        torch.where(point_valid[..., None], target_nocs, torch.zeros_like(target_nocs)),
        category,
        nocs_beta,
    )
    confidence = torch.sigmoid(confidence_logits)
    if torch.any(point_valid):
        loss_nocs = (
            nocs_values * confidence.detach() * point_valid.float()
        ).sum() / point_valid.float().sum().clamp_min(1.0)
    else:
        loss_nocs = zero
    loss_confidence = F.binary_cross_entropy_with_logits(
        confidence_logits,
        point_valid.to(confidence_logits.dtype),
    )

    predicted_camera = torch.bmm(
        target_nocs * pred_size[:, None, :],
        pred_rotation.transpose(1, 2),
    ) + pred_translation[:, None, :]
    point_distance = torch.linalg.vector_norm(
        predicted_camera - camera_points, dim=2
    )
    valid_alignment = point_valid & valid_pose[:, None]
    if torch.any(valid_alignment):
        loss_adds = (
            point_distance * valid_alignment.float()
        ).sum() / valid_alignment.float().sum().clamp_min(1.0)
    else:
        loss_adds = zero

    quality_margin = float(
        OmegaConf.select(cfg, "loss.joint_quality_margin", default=0.2)
    )
    ambiguous = (category == 0) | (category == 3)
    if torch.any(ambiguous):
        bottle_quality = end_points["bottle_pred_joint_quality_logit"]
        can_quality = end_points["can_pred_joint_quality_logit"]
        correct_quality = torch.where(category == 0, bottle_quality, can_quality)
        wrong_quality = torch.where(category == 0, can_quality, bottle_quality)
        loss_quality = F.relu(
            quality_margin - correct_quality[ambiguous] + wrong_quality[ambiguous]
        ).mean()
    else:
        loss_quality = zero

    total = (
        _configured_weight(cfg, "weight_joint_classification", 1.0)
        * loss_classification
        + _configured_weight(cfg, "weight_joint_nocs", 0.2) * loss_nocs
        + _configured_weight(cfg, "weight_joint_nocs_confidence", 0.01)
        * loss_confidence
        + _configured_weight(cfg, "weight_joint_translation", 1.0)
        * loss_translation
        + _configured_weight(cfg, "weight_joint_size_log", 0.5) * loss_size
        + _configured_weight(cfg, "weight_joint_adds", 0.2) * loss_adds
        + _configured_weight(cfg, "weight_joint_quality", 0.2) * loss_quality
    )
    predicted_category = class_logits.argmax(dim=1)
    detector_category = real_data.get("detector_category_label", category)
    detector_category = detector_category.to(category.device).reshape(-1).long()
    bottle_can = (category == 0) | (category == 3)
    corrected = predicted_category != detector_category
    helpful = corrected & (predicted_category == category)
    harmful = corrected & (detector_category == category)
    translation_error_cm = torch.linalg.vector_norm(
        pred_translation - gt_translation, dim=1
    ) * 100.0
    metrics = {
        "loss/joint_geometry": float(total.detach()),
        "loss/joint_classification": float(loss_classification.detach()),
        "loss/joint_nocs": float(loss_nocs.detach()),
        "loss/joint_nocs_confidence": float(loss_confidence.detach()),
        "loss/joint_translation": float(loss_translation.detach()),
        "loss/joint_size_log": float(loss_size.detach()),
        "loss/joint_adds": float(loss_adds.detach()),
        "loss/joint_quality": float(loss_quality.detach()),
        "joint/class_accuracy": float((predicted_category == category).float().mean()),
        "joint/detector_class_accuracy": float(
            (detector_category == category).float().mean()
        ),
        "joint/bottle_can_accuracy": float(
            (predicted_category[bottle_can] == category[bottle_can]).float().mean()
        ) if torch.any(bottle_can) else 0.0,
        "joint/correction_ratio": float(corrected.float().mean()),
        "joint/helpful_correction_ratio": float(helpful.float().mean()),
        "joint/harmful_correction_ratio": float(harmful.float().mean()),
        "joint/translation_mae_cm": float(translation_error_cm[valid_pose].mean())
        if torch.any(valid_pose) else 0.0,
        "joint/nocs_valid_ratio": float(point_valid.float().mean()),
        "joint/geometry_gate": float(
            end_points["pred_joint_geometry_gate"].detach().mean()
        ),
    }
    return total, metrics
