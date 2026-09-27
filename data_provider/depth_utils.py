"""Depth loading, encoding, and point-cloud projection."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


DINOV2_DEPTH_SCALE = 1000.0
DEPTH_PIPELINE_VERSION = "metric-depth-v2-full-image-projection"
VALID_DEPTH_SOURCES = {
    "legacy_estimated",
    "estimated",
    "sensor",
    "metric",
    "depth_png",
    "dinov2_base",
    "dinov2_large",
}
VALID_DEPTH_FEATURE_ENCODINGS = {"raw", "metres", "log1p"}
VALID_PROJECTION_MODES = {"full_image", "legacy_crop"}


def _decode_nocs_depth_png(depth_image: np.ndarray, scale: float) -> np.ndarray:
    if depth_image.ndim == 3:
        depth_u16 = (
            depth_image[:, :, 1].astype(np.uint16) * 256
            + depth_image[:, :, 2].astype(np.uint16)
        )
    else:
        depth_u16 = depth_image.astype(np.uint16)
    return depth_u16.astype(np.float32) / float(scale)


def _load_npy_metres(path: str, auto_scale: bool, scale: float) -> np.ndarray:
    depth = np.load(path).astype(np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth at {path}, got {depth.shape}")
    if auto_scale:
        valid = np.isfinite(depth) & (depth > 0.0)
        if np.any(valid) and float(np.nanmedian(depth[valid])) > 10.0:
            depth = depth / float(scale)
    return depth.astype(np.float32)


def depth_candidates(sample_path: str, source: str) -> List[Tuple[str, str]]:
    source = str(source).lower()
    if source in {"legacy_estimated", "estimated"}:
        return [(sample_path + "_depth_estimated.npy", "legacy_npy")]
    if source == "dinov2_base":
        return [
            (sample_path + "_dinov2_depth_u16.png", "metric_png"),
            (sample_path + "_dinov2_depth.npy", "metric_npy"),
        ]
    if source == "dinov2_large":
        return [
            (sample_path + "_dinov2_large_depth_u16.png", "metric_png"),
            (sample_path + "_dinov2_large_depth.npy", "metric_npy"),
        ]
    if source in {"sensor", "metric", "depth_png"}:
        return [
            (sample_path + "_depth.png", "nocs_png"),
            (sample_path + "_depth_gt.npy", "metric_npy_auto"),
            (sample_path + "_depth_real.npy", "metric_npy_auto"),
            (sample_path + "_depth.npy", "metric_npy_auto"),
        ]
    raise ValueError(
        f"Unsupported depth source={source!r}; expected one of {sorted(VALID_DEPTH_SOURCES)}"
    )


def resolve_depth_path(sample_path: str, source: str) -> Optional[Tuple[str, str]]:
    for path, depth_format in depth_candidates(sample_path, source):
        if os.path.isfile(path):
            return path, depth_format
    return None


def load_depth_metres(
        sample_path: str,
        source: str,
        png_scale: float = DINOV2_DEPTH_SCALE,
        required: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Load a depth source and return ``(depth, path)``.

    ``legacy_estimated`` is returned without claiming metric units and exists only
    for old-checkpoint compatibility. Every other source is returned in metres.
    """
    resolved = resolve_depth_path(sample_path, source)
    if resolved is None:
        if required:
            attempted = ", ".join(path for path, _ in depth_candidates(sample_path, source))
            raise FileNotFoundError(
                f"Missing depth source={source!r} for {sample_path}. Tried: {attempted}"
            )
        return None, None

    path, depth_format = resolved
    if depth_format == "legacy_npy":
        depth = np.load(path).astype(np.float32).squeeze()
    elif depth_format == "metric_png":
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_image is None or depth_image.ndim != 2 or depth_image.dtype != np.uint16:
            raise ValueError(
                f"Expected uint16 HxW metric depth PNG at {path}, got "
                f"{None if depth_image is None else (depth_image.shape, depth_image.dtype)}"
            )
        depth = depth_image.astype(np.float32) / float(png_scale)
    elif depth_format == "nocs_png":
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise OSError(f"Failed to read metric depth PNG: {path}")
        depth = _decode_nocs_depth_png(depth_image, scale=png_scale)
    elif depth_format == "metric_npy":
        depth = _load_npy_metres(path, auto_scale=False, scale=png_scale)
    elif depth_format == "metric_npy_auto":
        depth = _load_npy_metres(path, auto_scale=True, scale=png_scale)
    else:
        raise ValueError(f"Unsupported depth format={depth_format!r} for {path}")

    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth at {path}, got {depth.shape}")
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth, path


def mask_to_foreground(mask: np.ndarray) -> np.ndarray:
    """Convert SAM, NOCS instance, or binary prediction masks to 0/1 foreground."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    values = set(np.unique(mask).tolist())
    if not values:
        return np.zeros(mask.shape, dtype=np.uint8)
    if values == {255}:
        return np.zeros(mask.shape, dtype=np.uint8)
    # SAM files produced by this repo use object=1, background=255.
    if 255 in values and any(value not in {0, 255} for value in values):
        return (mask != 255).astype(np.uint8)
    # NOCS-style instance selections and predicted masks use zero background.
    return (mask > 0).astype(np.uint8)


def depth_to_point_cloud(
        depth: np.ndarray,
        cam_k: Sequence[float],
        bbox_xyxy: Optional[Sequence[int]] = None,
        projection_mode: str = "full_image",
) -> np.ndarray:
    """Backproject an HxW crop with the correct full-image pixel coordinates."""
    projection_mode = str(projection_mode)
    if projection_mode not in VALID_PROJECTION_MODES:
        raise ValueError(
            f"projection_mode must be one of {sorted(VALID_PROJECTION_MODES)}, "
            f"got {projection_mode!r}"
        )
    z = np.asarray(depth, dtype=np.float32)
    height, width = z.shape
    fx, fy, cx, cy = (float(value) for value in cam_k)
    x_offset = 0.0
    y_offset = 0.0
    if projection_mode == "full_image":
        if bbox_xyxy is None:
            raise ValueError("bbox_xyxy is required for full_image crop projection")
        x_offset = float(bbox_xyxy[0])
        y_offset = float(bbox_xyxy[1])

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32) + x_offset,
        np.arange(height, dtype=np.float32) + y_offset,
    )
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def resize_masked_depth(
        depth: np.ndarray,
        foreground: np.ndarray,
        output_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize depth without interpolating zero-valued background into the object."""
    foreground = (np.asarray(foreground) > 0).astype(np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    weighted = cv2.resize(
        depth * foreground,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )
    weights = cv2.resize(
        foreground,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )
    nearest_mask = cv2.resize(
        foreground.astype(np.uint8),
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)
    resized = np.zeros_like(weighted, dtype=np.float32)
    valid = (weights > 1.0e-6) & (nearest_mask > 0)
    resized[valid] = weighted[valid] / weights[valid]
    return resized.astype(np.float32), nearest_mask


def encode_depth_feature(
        depth: np.ndarray,
        foreground: np.ndarray,
        encoding: str = "log1p",
        min_depth_m: float = 0.05,
        max_depth_m: float = 5.0,
) -> np.ndarray:
    """Encode metric depth with a fixed transform shared by train and test."""
    encoding = str(encoding).lower()
    if encoding not in VALID_DEPTH_FEATURE_ENCODINGS:
        raise ValueError(
            f"Unsupported depth feature encoding={encoding!r}; "
            f"expected one of {sorted(VALID_DEPTH_FEATURE_ENCODINGS)}"
        )
    foreground = np.asarray(foreground) > 0
    valid = foreground & np.isfinite(depth) & (depth > 0.0)
    feature = np.zeros(np.asarray(depth).shape, dtype=np.float32)
    if not np.any(valid):
        return feature

    values = np.clip(
        np.asarray(depth, dtype=np.float32)[valid],
        float(min_depth_m),
        float(max_depth_m),
    )
    if encoding == "log1p":
        values = np.log1p(values)
    feature[valid] = values
    return feature


def augment_depth_consistently(
        depth: np.ndarray,
        foreground: np.ndarray,
        rng: np.random.Generator,
        scale_range: Sequence[float] = (1.0, 1.0),
        bias_m: float = 0.0,
        gaussian_std_m: float = 0.0,
) -> np.ndarray:
    """Perturb the depth map once so dense depth and projected points stay aligned."""
    output = np.asarray(depth, dtype=np.float32).copy()
    valid = (np.asarray(foreground) > 0) & np.isfinite(output) & (output > 0.0)
    if not np.any(valid):
        return output

    low, high = (float(scale_range[0]), float(scale_range[1]))
    scale = float(rng.uniform(min(low, high), max(low, high)))
    bias = float(rng.uniform(-abs(float(bias_m)), abs(float(bias_m))))
    values = output[valid] * scale + bias
    if gaussian_std_m > 0.0:
        values += rng.normal(0.0, float(gaussian_std_m), size=values.shape).astype(np.float32)
    output[valid] = np.maximum(values, 1.0e-6)
    return output.astype(np.float32)
