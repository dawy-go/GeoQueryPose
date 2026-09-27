from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


CATEGORY_NAMES = ("BG", "bottle", "bowl", "camera", "can", "laptop", "mug")
RECLASSIFIER_CLASS_IDS = (1, 4, 6)
RECLASSIFIER_CLASS_NAMES = tuple(CATEGORY_NAMES[index] for index in RECLASSIFIER_CLASS_IDS)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def bbox_iou_yxyx(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    y1 = max(float(box_a[0]), float(box_b[0]))
    x1 = max(float(box_a[1]), float(box_b[1]))
    y2 = min(float(box_a[2]), float(box_b[2]))
    x2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def match_boxes_class_agnostic(
    pred_bboxes: Sequence[Sequence[float]],
    gt_bboxes: Sequence[Sequence[float]],
    min_iou: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedily form one-to-one pairs in descending 2D IoU order."""
    candidates = []
    for pred_index, pred_bbox in enumerate(pred_bboxes):
        for gt_index, gt_bbox in enumerate(gt_bboxes):
            iou = bbox_iou_yxyx(pred_bbox, gt_bbox)
            if iou >= min_iou:
                candidates.append((iou, pred_index, gt_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    matches = []
    for iou, pred_index, gt_index in candidates:
        if pred_index in used_predictions or gt_index in used_ground_truth:
            continue
        used_predictions.add(pred_index)
        used_ground_truth.add(gt_index)
        matches.append((pred_index, gt_index, float(iou)))
    return sorted(matches, key=lambda item: item[1])


def detector_label_to_class_id(label: str) -> int:
    normalized = re.sub(r"[^a-z]+", " ", str(label).lower()).strip()
    tokens = set(normalized.split())
    for class_id, class_name in enumerate(CATEGORY_NAMES[1:], start=1):
        if class_name in tokens or normalized == class_name:
            return class_id
    return 0


def sample_relative_path(
    result_path: str | Path,
    result: Mapping[str, object],
) -> str:
    image_path = result.get("image_path")
    if image_path:
        normalized = str(image_path).replace("\\", "/")
        for suffix in ("_color.png", ".png", ".jpg", ".jpeg"):
            if normalized.lower().endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        marker = "/Real/"
        marker_index = normalized.lower().find(marker.lower())
        if marker_index >= 0:
            normalized = normalized[marker_index + len(marker) :]
        elif normalized.lower().startswith("real/"):
            normalized = normalized[len("real/") :]
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if re.fullmatch(r"(?:train|test)/scene_\d+/\d+", normalized):
            return normalized

    stem = Path(result_path).stem
    match = re.fullmatch(r"results_(train|test)_(scene_\d+)_(\d+)", stem)
    if not match:
        raise ValueError(
            f"Cannot resolve NOCS sample path from image_path={image_path!r} "
            f"or result name {Path(result_path).name!r}"
        )
    split, scene, frame = match.groups()
    return f"{split}/{scene}/{frame}"


def mask_to_foreground(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        mask = mask[:, :, 0]
    values = set(np.unique(mask).tolist())
    if not values or values == {255}:
        return np.zeros(mask.shape[:2], dtype=np.uint8)
    if 255 in values and any(value not in {0, 255} for value in values):
        return (mask != 255).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def _read_mask_image(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return mask


def detection_mask(
    result: Mapping[str, object],
    result_path: str | Path,
    pred_index: int,
    data_root: str | Path,
    sample_rel: str,
    image_shape: Sequence[int],
) -> np.ndarray:
    pred_masks = result.get("pred_masks")
    if pred_masks is not None:
        masks = np.asarray(pred_masks)
        if masks.ndim != 3:
            raise ValueError(f"pred_masks must be 3D, got {masks.shape}")
        if masks.shape[:2] == tuple(image_shape[:2]):
            if pred_index >= masks.shape[2]:
                raise IndexError(f"pred_mask index {pred_index} exceeds {masks.shape}")
            return mask_to_foreground(masks[:, :, pred_index])
        if masks.shape[1:] == tuple(image_shape[:2]):
            if pred_index >= masks.shape[0]:
                raise IndexError(f"pred_mask index {pred_index} exceeds {masks.shape}")
            return mask_to_foreground(masks[pred_index])
        raise ValueError(
            f"pred_masks shape {masks.shape} does not match image {tuple(image_shape[:2])}"
        )

    indexed_mask_path = result.get("pred_mask_path")
    if indexed_mask_path:
        indexed_path = Path(str(indexed_mask_path))
        if not indexed_path.is_absolute():
            indexed_path = Path(result_path).parent / indexed_path
        indexed = _read_mask_image(indexed_path)
        labels = np.asarray(
            result.get("pred_mask_labels", np.arange(len(result.get("pred_bboxes", []))) + 1)
        ).reshape(-1)
        if pred_index >= len(labels):
            raise IndexError(f"pred_mask_labels has no index {pred_index}")
        return (indexed == int(labels[pred_index])).astype(np.uint8)

    fallback_path = Path(data_root) / "Real" / f"{sample_rel}_mask_sam.png"
    return mask_to_foreground(_read_mask_image(fallback_path))


def read_sample_rgb(data_root: str | Path, sample_rel: str) -> np.ndarray:
    image_path = Path(data_root) / "Real" / f"{sample_rel}_color.png"
    rgb_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {image_path}")
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)


def expand_bbox_yxyx(
    bbox: Sequence[float],
    image_shape: Sequence[int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    y1, x1, y2, x2 = (float(value) for value in bbox)
    box_height = max(1.0, y2 - y1)
    box_width = max(1.0, x2 - x1)
    pad_y = box_height * max(0.0, float(padding_ratio))
    pad_x = box_width * max(0.0, float(padding_ratio))
    y1 = int(np.floor(np.clip(y1 - pad_y, 0, max(height - 1, 0))))
    x1 = int(np.floor(np.clip(x1 - pad_x, 0, max(width - 1, 0))))
    y2 = int(np.ceil(np.clip(y2 + pad_y, y1 + 1, height)))
    x2 = int(np.ceil(np.clip(x2 + pad_x, x1 + 1, width)))
    return y1, x1, y2, x2


def prepare_reclassifier_input(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox_yxyx: Sequence[float],
    image_size: int = 224,
    training: bool = False,
    padding_ratio: float = 0.08,
    background_keep: float = 0.20,
) -> torch.Tensor:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB input, got {rgb.shape}")
    foreground = mask_to_foreground(mask)
    if foreground.shape != rgb.shape[:2]:
        raise ValueError(f"Mask shape {foreground.shape} does not match RGB {rgb.shape[:2]}")

    y1, x1, y2, x2 = expand_bbox_yxyx(bbox_yxyx, rgb.shape[:2], padding_ratio)
    rgb_image = Image.fromarray(np.asarray(rgb[y1:y2, x1:x2], dtype=np.uint8), mode="RGB")
    mask_image = Image.fromarray((foreground[y1:y2, x1:x2] * 255).astype(np.uint8), mode="L")
    if not np.any(foreground[y1:y2, x1:x2]):
        mask_image = Image.new("L", rgb_image.size, color=255)

    if training:
        crop = RandomResizedCrop.get_params(
            rgb_image,
            scale=(0.82, 1.0),
            ratio=(0.85, 1.15),
        )
        rgb_image = TF.resized_crop(
            rgb_image,
            *crop,
            [image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask_image = TF.resized_crop(
            mask_image,
            *crop,
            [image_size, image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        if bool(torch.rand(()) < 0.5):
            rgb_image = TF.hflip(rgb_image)
            mask_image = TF.hflip(mask_image)
        rgb_image = ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03,
        )(rgb_image)
    else:
        rgb_image = TF.resize(
            rgb_image,
            [image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask_image = TF.resize(
            mask_image,
            [image_size, image_size],
            interpolation=InterpolationMode.NEAREST,
        )

    rgb_tensor = TF.to_tensor(rgb_image)
    mask_tensor = TF.to_tensor(mask_image).clamp(0.0, 1.0)
    mean_tensor = rgb_tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    keep = float(np.clip(background_keep, 0.0, 1.0))
    rgb_tensor = rgb_tensor * (mask_tensor + keep * (1.0 - mask_tensor))
    rgb_tensor = rgb_tensor + mean_tensor * (1.0 - keep) * (1.0 - mask_tensor)
    rgb_tensor = TF.normalize(rgb_tensor, IMAGENET_MEAN, IMAGENET_STD)
    mask_tensor = (mask_tensor - 0.5) / 0.5
    return torch.cat([rgb_tensor, mask_tensor], dim=0)


class NocsReclassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = len(RECLASSIFIER_CLASS_IDS),
        pretrained: bool = False,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")
        builder = getattr(models, backbone)
        weights = models.get_model_weights(builder).DEFAULT if pretrained else None
        network = builder(weights=weights)
        if not hasattr(network, "conv1") or not hasattr(network, "fc"):
            raise ValueError(f"Backbone {backbone!r} is not a ResNet-style model")

        original_conv = network.conv1
        mask_conv = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        with torch.no_grad():
            mask_conv.weight[:, :3].copy_(original_conv.weight)
            mask_conv.weight[:, 3:4].copy_(original_conv.weight.mean(dim=1, keepdim=True))
            if original_conv.bias is not None:
                mask_conv.bias.copy_(original_conv.bias)
        network.conv1 = mask_conv
        feature_dim = int(network.fc.in_features)
        network.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(feature_dim, num_classes))
        self.backbone_name = str(backbone)
        self.network = network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError(f"Expected [B,4,H,W] reclassifier input, got {tuple(inputs.shape)}")
        return self.network(inputs)


def confusion_matrix(
    targets: Iterable[int],
    predictions: Iterable[int],
    class_count: int,
) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        target = int(target)
        prediction = int(prediction)
        if 0 <= target < class_count and 0 <= prediction < class_count:
            matrix[target, prediction] += 1
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(matrix, dtype=np.int64)
    total = int(matrix.sum())
    recalls = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=matrix.sum(axis=1) > 0,
    )
    valid = matrix.sum(axis=1) > 0
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "balanced_accuracy": float(recalls[valid].mean()) if np.any(valid) else 0.0,
        "per_class_recall": recalls.tolist(),
        "confusion_matrix": matrix.tolist(),
        "count": total,
    }
