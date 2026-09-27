import os
import copy
import json
import pickle
import random
import re
import warnings
import glob
import hashlib
import math
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Sampler
from data_provider.point_cloud_denoiser import PointCloudDenoiser
from data_provider.depth_utils import (
    DEPTH_PIPELINE_VERSION,
    VALID_PROJECTION_MODES,
    augment_depth_consistently,
    depth_to_point_cloud as project_depth_to_point_cloud,
    encode_depth_feature,
    load_depth_metres,
    mask_to_foreground as canonical_mask_to_foreground,
    resize_masked_depth,
)
from modules.center_anchor import mask_center_from_crop


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DENOISE_CACHE_DIR = os.path.join(REPO_ROOT, "cache", "real275_denoised")
VALID_DENOISE_CACHE_MODES = {"off", "read_only", "read_write", "rebuild"}
VALID_RGB_ENHANCE_MODES = {"none", "brightness_contrast", "gamma", "clahe", "clahe_gamma"}


def load_pickle_ignoring_numpy_align_warning(file_obj):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            category=Warning,
        )
        return pickle.load(file_obj)


def enhance_rgb_crop(
        rgb_crop: np.ndarray,
        mode: str = "none",
        brightness: float = 0.0,
        contrast: float = 1.0,
        gamma: float = 1.0,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    mode = str(mode or "none")
    if mode == "none":
        return rgb_crop
    if mode not in VALID_RGB_ENHANCE_MODES:
        raise ValueError(f"Unsupported rgb enhance mode: {mode}")

    enhanced = rgb_crop.copy()
    if mode in {"brightness_contrast", "clahe_gamma"}:
        enhanced = np.clip(
            enhanced.astype(np.float32) * float(contrast) + float(brightness),
            0.0,
            255.0,
        ).astype(np.uint8)

    if mode in {"clahe", "clahe_gamma"}:
        tile_size = max(1, int(clahe_tile_grid_size))
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(tile_size, tile_size),
        )
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(
            cv2.merge([l_channel, a_channel, b_channel]),
            cv2.COLOR_LAB2RGB,
        )

    if mode in {"gamma", "clahe_gamma"}:
        gamma = max(float(gamma), 1.0e-6)
        enhanced = np.clip(
            np.power(enhanced.astype(np.float32) / 255.0, gamma) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

    return enhanced


def load_sample_list(base_dir: str, split_file: str = "train_list.txt") -> List[str]:
    split_path = os.path.join(base_dir, split_file)
    with open(split_path, "r", encoding="utf-8") as f:
        return [os.path.join(base_dir, line.strip()) for line in f if line.strip()]


def resolve_data_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def resolve_manifest_path(base_dir: str, path: str) -> str:
    """Resolve a manifest from either the repository or the NOCS data root."""
    expanded = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isabs(str(path)) or os.path.exists(expanded):
        return expanded

    candidates = [
        os.path.join(REPO_ROOT, str(path)),
        os.path.join(base_dir, str(path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def normalize_latent_code_dict(latent_codes: Dict) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in latent_codes.items():
        if isinstance(value, dict):
            normalized.update(normalize_latent_code_dict(value))
            continue
        normalized[str(key)] = torch.as_tensor(value).detach().cpu().clone()
    return normalized


def load_nocs_scene(sample_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    rgb_bgr = cv2.imread(sample_path + "_color.png", cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {sample_path}_color.png")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    depth_path = sample_path + "_depth_estimated.npy"
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Failed to read estimated depth file: {depth_path}")
    depth = np.load(depth_path).astype(np.float32)

    mask = cv2.imread(sample_path + "_mask_sam.png", cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read SAM mask: {sample_path}_mask_sam.png")

    with open(sample_path + "_label.pkl", "rb") as f:
        gts = pickle.load(f)

    return rgb, depth, mask, gts


def load_nocs_test_scene(sample_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    rgb_bgr = cv2.imread(sample_path + "_color.png", cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {sample_path}_color.png")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    depth_path = sample_path + "_depth_estimated.npy"
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Failed to read estimated depth file: {depth_path}")
    depth = np.load(depth_path).astype(np.float32)

    mask = cv2.imread(sample_path + "_mask_sam.png", cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read SAM mask: {sample_path}_mask_sam.png")

    bbox_path = sample_path + "_bbox.pkl"
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"Failed to read bbox file: {bbox_path}")
    with open(bbox_path, "rb") as f:
        bbox_data = pickle.load(f)

    return rgb, depth, mask, bbox_data


def _decode_metric_depth_image(depth_image: np.ndarray, png_scale: float = 1000.0) -> np.ndarray:
    if depth_image.ndim == 3:
        depth_u16 = depth_image[:, :, 1].astype(np.uint16) * 256 + depth_image[:, :, 2].astype(np.uint16)
    else:
        depth_u16 = depth_image.astype(np.uint16)
    return depth_u16.astype(np.float32) / float(png_scale)


def load_metric_depth_scene(
        sample_path: str,
        png_scale: float = 1000.0,
        npy_auto_scale: bool = True,
) -> Optional[np.ndarray]:
    del npy_auto_scale  # retained for API compatibility
    depth, _ = load_depth_metres(
        sample_path,
        source="sensor",
        png_scale=png_scale,
        required=False,
    )
    return depth


def resolve_test_sample_path(base_dir: str, sample_path: str) -> str:
    """
    Some list files may still store entries under `train/...`, while generated test
    predictions are actually placed under `test/...`. Try the original path first and
    then fall back to the test split path if needed.
    """
    bbox_path = sample_path + "_bbox.pkl"
    if os.path.exists(bbox_path):
        return sample_path

    relative_path = os.path.relpath(sample_path, base_dir)
    if relative_path.startswith(f"train{os.sep}") or relative_path.startswith("train/"):
        test_relative_path = relative_path.replace("train", "test", 1)
        fallback_sample_path = os.path.join(base_dir, test_relative_path)
        if os.path.exists(fallback_sample_path + "_bbox.pkl"):
            return fallback_sample_path

    return sample_path


def yxyx_to_xyxy(bbox: List[int], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    y1, x1, y2, x2 = bbox

    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y2 = int(np.clip(y2, 0, height - 1))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return x1, y1, x2, y2


def boxes_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    inter_w = min(x12, x22) - max(x11, x21)
    inter_h = min(y12, y22) - max(y11, y21)
    return inter_w > 0 and inter_h > 0


def find_non_overlapping_indices(
        bboxes_yxyx: List[List[int]],
        image_shape: Tuple[int, int],
) -> List[int]:
    boxes_xyxy = [yxyx_to_xyxy(bbox, image_shape) for bbox in bboxes_yxyx]
    valid_indices: List[int] = []

    for idx, box in enumerate(boxes_xyxy):
        has_overlap = any(
            boxes_overlap(box, other_box)
            for other_idx, other_box in enumerate(boxes_xyxy)
            if other_idx != idx
        )
        if not has_overlap:
            valid_indices.append(idx)

    return valid_indices


def normalize_rel_path(path: str, base_dir: str) -> str:
    rel_path = os.path.relpath(path, base_dir)
    return rel_path.replace("\\", "/")


def file_fingerprint(path: str) -> str:
    stat = os.stat(path)
    return f"{os.path.basename(path)}:{stat.st_size}:{stat.st_mtime_ns}"


def real275_denoise_cache_name(
        sample_path: str,
        obj_idx: int,
        bbox_yxyx,
        data_dir: str,
        sample_num: int,
        depth_path: str,
        mask_path: str,
        projection_mode: str,
) -> str:
    parts = [
        normalize_rel_path(sample_path, data_dir),
        f"obj={obj_idx}",
        "bbox=" + ",".join(str(int(v)) for v in bbox_yxyx),
        f"sample_num={sample_num}",
        file_fingerprint(depth_path),
        file_fingerprint(mask_path),
        file_fingerprint(sample_path + "_label.pkl"),
        f"pipeline={DEPTH_PIPELINE_VERSION}",
        f"projection={projection_mode}",
        "denoise=statistical_outlier_nb20_std2.0",
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    scene = os.path.basename(os.path.dirname(sample_path))
    image_id = os.path.basename(sample_path)
    return f"{scene}_{image_id}_obj{obj_idx}_{digest}.npz"


def real275_denoise_cache_path(
        sample_path: str,
        obj_idx: int,
        bbox_yxyx,
        data_dir: str,
        sample_num: int,
        cache_dir: str,
        depth_path: str,
        mask_path: str,
        projection_mode: str,
) -> str:
    return os.path.join(
        cache_dir,
        real275_denoise_cache_name(
            sample_path,
            obj_idx,
            bbox_yxyx,
            data_dir,
            sample_num,
            depth_path,
            mask_path,
            projection_mode,
        ),
    )


def real275_normal_cache_path(point_cache_path: str) -> str:
    stem, _ = os.path.splitext(point_cache_path)
    return stem + ".normals.npz"


def atomic_save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        npz_tmp_path = tmp_path if tmp_path.endswith(".npz") else tmp_path + ".npz"
        os.replace(npz_tmp_path, path)
    finally:
        for candidate in (tmp_path, tmp_path + ".npz"):
            if os.path.exists(candidate):
                os.remove(candidate)


def load_point_branch_cache(cache_path: str) -> Dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "point_cloud_filtered": data["point_cloud_filtered"].astype(np.float32),
            "rgb_points_filtered": data["rgb_points_filtered"].astype(np.float32),
            "pts_sampled": data["pts_sampled"].astype(np.float32),
            "rgb_points_sampled": data["rgb_points_sampled"].astype(np.float32),
            "sampled_pixel_indices": data["sampled_pixel_indices"].astype(np.int64),
        }


def estimate_normals_for_points_np(points: np.ndarray, knn: int = 30, batch_size: int = 1024) -> np.ndarray:
    points_tensor = torch.as_tensor(points, dtype=torch.float32)
    num_points = points_tensor.shape[0]
    if num_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    effective_knn = min(int(knn), int(num_points))
    idx_list = []
    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        dist = torch.cdist(points_tensor[start:end], points_tensor)
        _, idx = dist.topk(k=effective_knn, largest=False, dim=-1)
        idx_list.append(idx)

    idx = torch.cat(idx_list, dim=0)
    neighbors = points_tensor[idx]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    denom = max(effective_knn - 1, 1)
    cov = torch.matmul(centered.transpose(-2, -1), centered) / float(denom)
    _, eigenvectors = torch.linalg.eigh(cov)
    normals = eigenvectors[:, :, 0]
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1
    return normals.numpy().astype(np.float32)


def load_normal_cache(cache_path: str) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as data:
        return data["normals_sampled"].astype(np.float32)


def save_normal_cache(cache_path: str, normals_sampled: np.ndarray) -> None:
    atomic_save_npz(cache_path, {"normals_sampled": normals_sampled.astype(np.float32)})


def build_normals_for_point_branch(point_branch: Dict[str, np.ndarray]) -> np.ndarray:
    return estimate_normals_for_points_np(point_branch["pts_sampled"].astype(np.float32))


def save_point_branch_cache(
        cache_path: str,
        point_branch: Dict[str, np.ndarray],
        bbox_yxyx,
        obj_idx: int,
        sample_path: str,
) -> None:
    arrays = {
        "point_cloud_filtered": point_branch["point_cloud_filtered"].astype(np.float32),
        "rgb_points_filtered": point_branch["rgb_points_filtered"].astype(np.float32),
        "sampled_pixel_indices": point_branch["sampled_pixel_indices"].astype(np.int64),
        "pts_sampled": point_branch["pts_sampled"].astype(np.float32),
        "rgb_points_sampled": point_branch["rgb_points_sampled"].astype(np.float32),
        "bbox_yxyx": np.asarray(bbox_yxyx, dtype=np.int32),
        "object_index": np.asarray([obj_idx], dtype=np.int64),
        "sample_path": np.asarray([sample_path]),
    }
    atomic_save_npz(cache_path, arrays)


def build_real_test_gt_path(sample_path: str, gt_root: str) -> str:
    scene_name = os.path.basename(os.path.dirname(sample_path))
    image_name = os.path.basename(sample_path)
    return os.path.join(gt_root, f"results_real_test_{scene_name}_{image_name}.pkl")


def resolve_real275_result_image_path(base_dir: str, image_path: str) -> str:
    normalized = image_path.replace("\\", "/")
    prefixes = [
        ("data/real/", "Real/"),
        ("data/Real/", "Real/"),
    ]
    for prefix, replacement in prefixes:
        if normalized.startswith(prefix):
            normalized = replacement + normalized[len(prefix):]
            break
    return os.path.join(base_dir, *normalized.split("/"))


def sample_path_from_real275_result_name(base_dir: str, result_path: str) -> str:
    match = re.match(
        r"results_(?:real_)?test_scene_(\d+)_(\d+)\.pkl$",
        os.path.basename(result_path),
    )
    if not match:
        raise KeyError(
            f"Missing image_path and cannot infer REAL275 sample path from: {result_path}"
        )
    scene_id, image_id = match.groups()
    return os.path.join(base_dir, "Real", "test", f"scene_{int(scene_id)}", image_id)


def mask_to_foreground(mask: np.ndarray) -> np.ndarray:
    return canonical_mask_to_foreground(mask)


def load_fallback_real275_pred_masks(sample_path: str, instance_count: int) -> np.ndarray:
    mask_path = sample_path + "_mask_sam.png"
    scene_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if scene_mask is None:
        raise FileNotFoundError(f"Missing pred_masks and fallback SAM mask: {mask_path}")
    foreground = mask_to_foreground(scene_mask)
    return np.repeat(foreground[:, :, None], instance_count, axis=2)


def load_real275_result_pred_masks(
        result_data: Dict,
        result_path: str,
        sample_path: str,
        image_shape: Tuple[int, int],
        instance_count: int,
) -> np.ndarray:
    pred_masks = result_data.get("pred_masks")
    if pred_masks is not None:
        masks = np.asarray(pred_masks)
        if masks.ndim != 3:
            raise ValueError(f"pred_masks must be 3D, got {masks.shape}")
        if masks.shape[:2] == tuple(image_shape):
            return masks
        if masks.shape[1:] == tuple(image_shape):
            return np.transpose(masks, (1, 2, 0))
        raise ValueError(
            f"pred_masks shape {masks.shape} does not match image {tuple(image_shape)}"
        )

    indexed_mask_value = result_data.get("pred_mask_path")
    if indexed_mask_value:
        indexed_mask_path = os.path.expanduser(str(indexed_mask_value))
        if not os.path.isabs(indexed_mask_path):
            indexed_mask_path = os.path.join(
                os.path.dirname(os.path.abspath(result_path)),
                indexed_mask_path,
            )
        indexed_mask = cv2.imread(indexed_mask_path, cv2.IMREAD_UNCHANGED)
        if indexed_mask is None:
            raise FileNotFoundError(
                f"Missing indexed prediction mask: {indexed_mask_path}"
            )
        if indexed_mask.ndim == 3:
            indexed_mask = indexed_mask[:, :, 0]
        if indexed_mask.shape != tuple(image_shape):
            raise ValueError(
                f"Indexed prediction mask shape {indexed_mask.shape} does not match "
                f"image {tuple(image_shape)}: {indexed_mask_path}"
            )
        labels = np.asarray(
            result_data.get(
                "pred_mask_labels",
                np.arange(1, instance_count + 1, dtype=np.int64),
            ),
            dtype=np.int64,
        ).reshape(-1)
        return np.stack(
            [(indexed_mask == int(label)).astype(np.uint8) for label in labels],
            axis=2,
        )

    return load_fallback_real275_pred_masks(sample_path, instance_count)


def decompose_srt(srt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    rotation_scale = srt[:3, :3].astype(np.float32)
    translation = srt[:3, 3].astype(np.float32)

    scale = float(np.cbrt(np.linalg.det(rotation_scale)))
    if abs(scale) < 1e-8:
        scale = 1.0
    rotation = rotation_scale / scale
    return rotation.astype(np.float32), translation.astype(np.float32), scale


def depth_to_point_cloud(
        depth: np.ndarray,
        cam_k: List[float],
        bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
        projection_mode: str = "full_image",
) -> np.ndarray:
    return project_depth_to_point_cloud(
        depth,
        cam_k,
        bbox_xyxy=bbox_xyxy,
        projection_mode=projection_mode,
    )


def sample_point_map_at_pixels(point_map: np.ndarray, pixel_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(pixel_indices) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=bool)
    height, width = point_map.shape[:2]
    rows = pixel_indices[:, 0].astype(np.int64)
    cols = pixel_indices[:, 1].astype(np.int64)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows_safe = np.clip(rows, 0, max(height - 1, 0))
    cols_safe = np.clip(cols, 0, max(width - 1, 0))
    points = point_map[rows_safe, cols_safe].astype(np.float32)
    valid = in_bounds & np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    points[~valid] = 0.0
    return points, valid


def sample_fps_indices(points: np.ndarray, sample_num: int) -> np.ndarray:
    num_points = points.shape[0]
    if num_points == 0:
        return np.empty((0,), dtype=np.int64)

    if num_points == 1:
        return np.zeros((sample_num,), dtype=np.int64)

    selected = np.zeros((min(sample_num, num_points),), dtype=np.int64)
    distances = np.full((num_points,), np.inf, dtype=np.float32)
    farthest = 0

    for idx in range(selected.shape[0]):
        selected[idx] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))

    if num_points >= sample_num:
        return selected

    extra = np.random.choice(selected, sample_num - num_points, replace=True)
    return np.concatenate([selected, extra]).astype(np.int64)


def filter_and_sample_points(
        point_cloud_crop: np.ndarray,
        rgb_crop: np.ndarray,
        mask_crop: np.ndarray,
        sample_num: int,
) -> Dict[str, np.ndarray]:
    valid_mask = mask_crop.astype(bool) & np.isfinite(point_cloud_crop[:, :, 2]) & (point_cloud_crop[:, :, 2] > 0)
    valid_indices_2d = np.argwhere(valid_mask)
    if len(valid_indices_2d) == 0:
        return {
            "point_cloud_filtered" : np.empty((0, 3), dtype=np.float32),
            "rgb_points_filtered"  : np.empty((0, 3), dtype=np.float32),
            "pts_sampled"          : np.empty((0, 3), dtype=np.float32),
            "rgb_points_sampled"   : np.empty((0, 3), dtype=np.float32),
            "sampled_pixel_indices": np.empty((0, 2), dtype=np.int64),
        }

    points = point_cloud_crop[valid_mask].astype(np.float32)
    colors = (rgb_crop[valid_mask].astype(np.float32) / 255.0).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = PointCloudDenoiser.denoise_point_cloud_statistical_outlier(pcd)

    filtered_points = np.asarray(pcd.points, dtype=np.float32)
    filtered_colors = np.asarray(pcd.colors, dtype=np.float32)
    if len(filtered_points) == 0:
        filtered_points = points
        filtered_colors = colors
        filtered_pixel_indices = valid_indices_2d.astype(np.int64)
    else:
        rounded_point_to_indices: Dict[Tuple[float, float, float], List[int]] = {}
        for idx, point in enumerate(points):
            key = tuple(np.round(point, 6))
            rounded_point_to_indices.setdefault(key, []).append(idx)

        matched_indices: List[int] = []
        for point in filtered_points:
            key = tuple(np.round(point, 6))
            candidates = rounded_point_to_indices.get(key)
            if candidates:
                matched_indices.append(candidates.pop(0))

        if len(matched_indices) != len(filtered_points):
            filtered_points = points
            filtered_colors = colors
            filtered_pixel_indices = valid_indices_2d.astype(np.int64)
        else:
            filtered_pixel_indices = valid_indices_2d[np.asarray(matched_indices, dtype=np.int64)].astype(np.int64)

    fps_indices = sample_fps_indices(filtered_points, sample_num)
    return {
        "point_cloud_filtered" : filtered_points.astype(np.float32),
        "rgb_points_filtered"  : filtered_colors.astype(np.float32),
        "pts_sampled"          : filtered_points[fps_indices].astype(np.float32),
        "rgb_points_sampled"   : filtered_colors[fps_indices].astype(np.float32),
        "sampled_pixel_indices": filtered_pixel_indices[fps_indices].astype(np.int64),
    }


def crop_scene_fields(
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
) -> Dict[str, np.ndarray]:
    x1, y1, x2, y2 = bbox_xyxy

    rgb_crop = rgb[y1:y2, x1:x2].copy()
    depth_crop = depth[y1:y2, x1:x2].copy()
    crop_mask = mask_to_foreground(mask[y1:y2, x1:x2])

    return {
        "rgb_crop"  : rgb_crop,
        "depth_crop": depth_crop,
        "mask_crop" : crop_mask,
    }


def resize_aligned_modalities(
        rgb_crop: np.ndarray,
        depth_crop: np.ndarray,
        mask_crop: np.ndarray,
        img_size: int,
        depth_feature_encoding: str = "raw",
        depth_feature_min_m: float = 0.05,
        depth_feature_max_m: float = 5.0,
) -> Dict[str, np.ndarray]:
    rgb_masked = rgb_crop.copy()
    rgb_masked[mask_crop == 0] = 0
    rgb_resized = cv2.resize(
        rgb_masked,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )

    depth_metric_resized, mask_resized = resize_masked_depth(
        depth_crop,
        mask_crop,
        img_size,
    )
    depth_resized = encode_depth_feature(
        depth_metric_resized,
        mask_resized,
        encoding=depth_feature_encoding,
        min_depth_m=depth_feature_min_m,
        max_depth_m=depth_feature_max_m,
    )

    return {
        "rgb_resized"  : rgb_resized,
        "depth_resized": depth_resized,
        "depth_metric_resized": depth_metric_resized,
        "mask_resized" : mask_resized,
    }


def _stack_tensor_field(batch: List[Dict], key: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack(
        [torch.as_tensor(sample[key], dtype=dtype) for sample in batch],
        dim=0,
    )


def _copy_list_field(batch: List[Dict], key: str) -> List:
    return [sample.get(key) for sample in batch]


def _model_name_candidates(model_name: str) -> List[str]:
    normalized = str(model_name).replace("\\", os.sep)
    basename = os.path.basename(normalized)
    stem, _ = os.path.splitext(basename)
    return [str(model_name), basename, stem]


def _safe_indexed_value(values, index: int, default):
    if values is None:
        return default
    try:
        if index < len(values):
            return values[index]
    except TypeError:
        return default
    return default


def batch_to_device(batch: Dict, device) -> Dict:
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


def collate_fn(batch: List[Dict]) -> Dict:
    collated = {
        "rgb"              : torch.stack([sample["rgb"] for sample in batch], dim=0),
        "depth"            : _stack_tensor_field(batch, "depth", torch.float32),
        "pts"              : _stack_tensor_field(batch, "pts", torch.float32),
        "choose"           : _stack_tensor_field(batch, "choose", torch.long),
        "rgb_points"       : _stack_tensor_field(batch, "rgb_points", torch.float32),
        "category_label"   : _stack_tensor_field(batch, "category_label", torch.long),
        "rotation_label"   : _stack_tensor_field(batch, "rotation_label", torch.float32),
        "translation_label": _stack_tensor_field(batch, "translation_label", torch.float32),
    }

    optional_tensor_fields = {
        "scene_rgb"  : torch.float32,
        "scene_metric_depth": torch.float32,
        "scene_object_mask": torch.bool,
        "bbox"       : torch.long,
        "class_id"   : torch.long,
        "instance_id": torch.long,
        "size_label" : torch.float32,
        "model"      : torch.float32,
        "qo"         : torch.float32,
        "shape_code" : torch.float32,
        "score"      : torch.float32,
        "scale_label": torch.float32,
        "gt_sRT"     : torch.float32,
        "normals"    : torch.float32,
        "metric_depth": torch.float32,
        "metric_pts": torch.float32,
        "metric_depth_valid": torch.bool,
        "pts_local": torch.float32,
        "pts_metric": torch.float32,
        "pts_metric_valid": torch.bool,
        "shape_code_valid": torch.bool,
        "source_id": torch.long,
        "mask_center_uv": torch.float32,
        "mask_center_valid": torch.bool,
        "mask_area_ratio": torch.float32,
        "scene_mask_fallback_used": torch.bool,
        "cam_k": torch.float32,
        "image_hw": torch.float32,
        "detection_pred_class_id": torch.long,
        "detection_score": torch.float32,
        "detection_bbox_iou": torch.float32,
        "detector_category_label": torch.long,
        "detector_score": torch.float32,
    }
    for key, dtype in optional_tensor_fields.items():
        if all(key in sample for sample in batch):
            collated[key] = _stack_tensor_field(batch, key, dtype)

    list_fields = [
        "sample_path",
        "object_index",
        "bbox_format",
        "model_name",
        "scene_model_list",
        "source",
        "label_name",
        "gt_image_path",
        "rgb_crop",
        "depth_crop",
        "metric_depth_crop",
        "depth_source",
        "depth_path",
        "metric_depth_source",
        "metric_depth_path",
        "projection_mode",
        "mask_crop",
        "point_cloud_crop",
        "point_cloud_filtered",
        "rgb_points_filtered",
        "detection_result_path",
        "detection_manifest_line",
        "detection_index",
    ]
    for key in list_fields:
        if any(key in sample for sample in batch):
            collated[key] = _copy_list_field(batch, key)

    return collated


def test_collate_fn(batch: List[Dict]) -> Dict:
    if len(batch) != 1:
        raise ValueError("Real275TestDataset uses variable instances per image; keep test batch_size=1.")

    sample = batch[0]
    collated = {
        "rgb"                 : sample["rgb"].unsqueeze(0),
        "depth"               : torch.as_tensor(sample["depth"], dtype=torch.float32).unsqueeze(0),
        "metric_depth"        : torch.as_tensor(
            sample.get("metric_depth", sample["depth"]), dtype=torch.float32
        ).unsqueeze(0),
        "pts"                 : torch.as_tensor(sample["pts"], dtype=torch.float32).unsqueeze(0),
        "pts_local"           : torch.as_tensor(sample.get("pts_local", sample["pts"]), dtype=torch.float32).unsqueeze(0),
        "pts_metric"          : torch.as_tensor(sample.get("pts_metric", sample["pts"]), dtype=torch.float32).unsqueeze(0),
        "pts_metric_valid"    : torch.as_tensor(
            sample.get("pts_metric_valid", np.isfinite(sample["pts"]).all(axis=1) & (sample["pts"][:, 2] > 0.0)),
            dtype=torch.bool,
        ).unsqueeze(0),
        "choose"              : torch.as_tensor(sample["choose"], dtype=torch.long).unsqueeze(0),
        "rgb_points"          : torch.as_tensor(sample["rgb_points"], dtype=torch.float32).unsqueeze(0),
        "category_label"      : torch.as_tensor(sample["category_label"], dtype=torch.long).unsqueeze(0),
        "bbox"                : torch.as_tensor(sample["bbox"], dtype=torch.float32).unsqueeze(0),
        "score"               : torch.as_tensor(sample["score"], dtype=torch.float32).unsqueeze(0),
        "gt_class_ids"        : torch.as_tensor(sample["gt_class_ids"], dtype=torch.long).unsqueeze(0),
        "gt_bboxes"           : torch.as_tensor(sample["gt_bboxes"], dtype=torch.float32).unsqueeze(0),
        "gt_RTs"              : torch.as_tensor(sample["gt_RTs"], dtype=torch.float32).unsqueeze(0),
        "gt_scales"           : torch.as_tensor(sample["gt_scales"], dtype=torch.float32).unsqueeze(0),
        "gt_handle_visibility": torch.as_tensor(sample["gt_handle_visibility"], dtype=torch.float32).unsqueeze(0),
        "pred_class_ids"      : torch.as_tensor(sample["pred_class_ids"], dtype=torch.long).unsqueeze(0),
        "pred_bboxes"         : torch.as_tensor(sample["pred_bboxes"], dtype=torch.float32).unsqueeze(0),
        "pred_scores"         : torch.as_tensor(sample["pred_scores"], dtype=torch.float32).unsqueeze(0),
        "detector_category_label": torch.as_tensor(
            sample["category_label"], dtype=torch.long
        ).unsqueeze(0),
        "detector_score"      : torch.as_tensor(
            sample["pred_scores"], dtype=torch.float32
        ).unsqueeze(0),
        "mask_center_uv"      : torch.as_tensor(sample["mask_center_uv"], dtype=torch.float32).unsqueeze(0),
        "mask_center_valid"   : torch.as_tensor(sample["mask_center_valid"], dtype=torch.bool).unsqueeze(0),
        "mask_area_ratio"     : torch.as_tensor(sample["mask_area_ratio"], dtype=torch.float32).unsqueeze(0),
        "cam_k"               : torch.as_tensor(sample["cam_k"], dtype=torch.float32).unsqueeze(0),
        "image_hw"            : torch.as_tensor(sample["image_hw"], dtype=torch.float32).unsqueeze(0),
        "sample_path"         : [sample.get("sample_path")],
        "object_index"        : [sample.get("object_index")],
        "bbox_format"         : [sample.get("bbox_format")],
        "label_name"          : [sample.get("label_name")],
        "gt_image_path"       : [sample.get("gt_image_path")],
        "ori_img"             : [sample.get("ori_img")],
        "depth_source"        : [sample.get("depth_source")],
        "depth_path"          : [sample.get("depth_path")],
        "metric_depth_source" : [sample.get("metric_depth_source")],
        "metric_depth_path"   : [sample.get("metric_depth_path")],
        "projection_mode"     : [sample.get("projection_mode")],
    }
    if "scene_rgb" in sample:
        collated["scene_rgb"] = torch.as_tensor(
            sample["scene_rgb"], dtype=torch.float32
        ).unsqueeze(0)
    if "scene_metric_depth" in sample:
        collated["scene_metric_depth"] = torch.as_tensor(
            sample["scene_metric_depth"], dtype=torch.float32
        ).unsqueeze(0)
    if "scene_object_mask" in sample:
        collated["scene_object_mask"] = torch.as_tensor(
            sample["scene_object_mask"], dtype=torch.bool
        ).unsqueeze(0)
    return collated


def build_rgb_only_point_branch(sample_num: int) -> Dict[str, np.ndarray]:
    """Compatibility tensors for RGB-only models that never consume points."""
    count = max(1, int(sample_num))
    return {
        "point_cloud_filtered": np.zeros((count, 3), dtype=np.float32),
        "rgb_points_filtered": np.zeros((count, 3), dtype=np.float32),
        "pts_sampled": np.zeros((count, 3), dtype=np.float32),
        "rgb_points_sampled": np.zeros((count, 3), dtype=np.float32),
        "sampled_pixel_indices": np.zeros((count, 2), dtype=np.int64),
    }


class Real275Dataset(Dataset):
    """
    Read a NOCS-style dataset list and return one object instance per sample.

    Notes:
    - All valid object instances in one image are used before moving to the next image.
    - The crop mask comes from `*_mask_sam.png`, so if multiple objects fall inside the
      selected bbox, all foreground pixels inside that bbox are retained.
    """

    def __init__(
        self,
        base_dir: str,
        cam_k: Optional[List[float]] = None,
        seed: Optional[int] = None,
        model_path: str = "obj_models/real_train.pkl",
        shape_code_path: Optional[str] = "obj_models/latent_code/real_train.pth",
        shape_code_required: bool = True,
        shape_code_dim: int = 256,
        source_dir: str = "Real",
        split_file: str = "train_list.txt",
        source_name: str = "real",
        source_id: int = 0,
        depth_mode: str = "estimated",
        mask_mode: str = "sam",
        img_size: int = 224,
        sample_num: int = 2048,
        batch_size: Optional[int] = None,
        num_mini_batch: Optional[int] = None,
        image_shuffle: bool = True,
        image_shape: Tuple[int, int] = (480, 640),
        denoise_cache_enable: bool = False,
        denoise_cache_dir: Optional[str] = None,
        denoise_cache_mode: str = "read_write",
        skip_bad_samples: bool = True,
        bad_sample_retry: int = 32,
        metric_depth_enable: bool = False,
        metric_depth_required: bool = False,
        metric_depth_png_scale: float = 1000.0,
        metric_depth_source: str = "sensor",
        projection_mode: str = "full_image",
        depth_feature_encoding: str = "raw",
        depth_feature_min_m: float = 0.05,
        depth_feature_max_m: float = 5.0,
        augmentation: Optional[Dict] = None,
        canonicalize_symmetry_rotation: bool = False,
        scene_mask_fallback_to_bbox: bool = False,
        scene_mask_min_pixels: int = 1,
        detection_manifest: Optional[str] = None,
        detection_results_dir: Optional[str] = None,
        detection_split: str = "train",
        detection_min_iou: float = 0.5,
        detection_require_class_match: bool = True,
    ) -> None:
        self.data_dir = base_dir
        self.base_dir = os.path.join(base_dir, source_dir)
        self.split_file = split_file
        self.sample_paths = load_sample_list(base_dir=self.base_dir , split_file=self.split_file)
        self.detection_manifest_path = (
            resolve_manifest_path(base_dir, detection_manifest)
            if detection_manifest
            else None
        )
        self.detection_results_dir = (
            os.path.abspath(os.path.expanduser(detection_results_dir))
            if detection_results_dir and os.path.isabs(detection_results_dir)
            else (
                os.path.abspath(os.path.join(base_dir, detection_results_dir))
                if detection_results_dir
                else None
            )
        )
        self.detection_split = str(detection_split)
        self.detection_min_iou = float(detection_min_iou)
        self.detection_require_class_match = bool(detection_require_class_match)
        self.detection_rows = self._load_detection_manifest()
        self.uses_detection_manifest = bool(self.detection_rows)
        self.ordered_detection_indices = list(range(len(self.detection_rows)))
        self.dataset_length = self._resolve_dataset_length(batch_size, num_mini_batch)
        self.cam_k = cam_k or [591.0125, 590.16775, 322.525, 244.11084]
        self.seed = seed
        self.source_name = str(source_name)
        self.source_id = int(source_id)
        self.depth_mode = str(depth_mode)
        self.mask_mode = str(mask_mode)
        self.shape_code_required = bool(shape_code_required)
        self.shape_code_dim = int(shape_code_dim)
        self.image_shuffle = image_shuffle
        self.image_shape = image_shape
        self.epoch = 0
        self.ordered_sample_paths = list(self.sample_paths)
        self.rng = random.Random(seed)
        self.img_size = img_size
        self.sample_num = sample_num
        self.models = {}
        self.denoise_cache_enable = bool(denoise_cache_enable)
        cache_dir = denoise_cache_dir or DEFAULT_DENOISE_CACHE_DIR
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(REPO_ROOT, cache_dir)
        self.denoise_cache_dir = os.path.abspath(cache_dir)
        self.denoise_cache_mode = str(denoise_cache_mode)
        self.skip_bad_samples = bool(skip_bad_samples)
        self.bad_sample_retry = max(1, int(bad_sample_retry))
        self.metric_depth_enable = bool(metric_depth_enable)
        self.metric_depth_required = bool(metric_depth_required)
        self.metric_depth_png_scale = float(metric_depth_png_scale)
        self.metric_depth_source = str(metric_depth_source).lower()
        self.projection_mode = str(projection_mode)
        if self.projection_mode not in VALID_PROJECTION_MODES:
            raise ValueError(
                f"projection_mode must be one of {sorted(VALID_PROJECTION_MODES)}, "
                f"got {self.projection_mode!r}"
            )
        self.depth_feature_encoding = str(depth_feature_encoding)
        self.depth_feature_min_m = float(depth_feature_min_m)
        self.depth_feature_max_m = float(depth_feature_max_m)
        self.augmentation = dict(augmentation or {})
        self.augmentation_enable = bool(self.augmentation.get("enable", False))
        color_strength = self.augmentation.get("color_jitter", [0.2, 0.2, 0.2, 0.05])
        if len(color_strength) != 4:
            raise ValueError("augmentation.color_jitter must contain [brightness, contrast, saturation, hue]")
        self.color_jitter = transforms.ColorJitter(*[float(value) for value in color_strength])
        self.canonicalize_symmetry_rotation = bool(canonicalize_symmetry_rotation)
        self.scene_mask_fallback_to_bbox = bool(scene_mask_fallback_to_bbox)
        self.scene_mask_min_pixels = max(1, int(scene_mask_min_pixels))
        self.sym_ids = {0, 1, 3}
        if self.denoise_cache_mode not in VALID_DENOISE_CACHE_MODES:
            raise ValueError(
                f"denoise_cache_mode must be one of {sorted(VALID_DENOISE_CACHE_MODES)}, "
                f"got {self.denoise_cache_mode}"
            )
        if self.denoise_cache_mode == "off":
            self.denoise_cache_enable = False
        if self.denoise_cache_enable and self.denoise_cache_mode in {"read_write", "rebuild"}:
            os.makedirs(self.denoise_cache_dir, exist_ok=True)
        self.set_epoch(0)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        full_model_path = resolve_data_path(base_dir, model_path)
        self.latent_code_path = resolve_data_path(base_dir, shape_code_path)
        with open(full_model_path, "rb") as f:
            self.models.update(load_pickle_ignoring_numpy_align_warning(f))
        self.latent_codes = {}
        if self.latent_code_path is not None and os.path.exists(self.latent_code_path):
            latent_codes = torch.load(self.latent_code_path, map_location="cpu")
            if not isinstance(latent_codes, dict):
                raise TypeError(f"Latent code file must contain a dict: {self.latent_code_path}")
            self.latent_codes = normalize_latent_code_dict(latent_codes)
        elif self.shape_code_required:
            raise FileNotFoundError(f"Shape code file not found: {self.latent_code_path}")

    def _load_detection_manifest(self) -> List[Dict]:
        if self.detection_manifest_path is None:
            if self.detection_results_dir is not None:
                raise ValueError(
                    "train_dataset.real.detection_results_dir requires detection_manifest"
                )
            return []
        if self.detection_results_dir is None:
            raise ValueError(
                "train_dataset.real.detection_manifest requires detection_results_dir"
            )
        if not os.path.isfile(self.detection_manifest_path):
            raise FileNotFoundError(
                f"Detection manifest not found: {self.detection_manifest_path}"
            )
        if not os.path.isdir(self.detection_results_dir):
            raise FileNotFoundError(
                f"Detection results directory not found: {self.detection_results_dir}"
            )
        if not 0.0 <= self.detection_min_iou <= 1.0:
            raise ValueError("detection_min_iou must be in [0, 1]")

        rows: List[Dict] = []
        with open(self.detection_manifest_path, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {self.detection_manifest_path}:{line_number}"
                    ) from exc

                required = {
                    "sample_rel",
                    "result_path",
                    "pred_index",
                    "pred_bbox_yxyx",
                    "pred_class_id",
                    "gt_index",
                    "gt_class_id",
                    "bbox_iou",
                }
                missing = sorted(required.difference(row))
                if missing:
                    raise KeyError(
                        f"Detection manifest row {line_number} is missing: {missing}"
                    )
                if self.detection_split.lower() != "all" and str(
                    row.get("split", "train")
                ) != self.detection_split:
                    continue
                if float(row["bbox_iou"]) < self.detection_min_iou:
                    continue
                pred_class_id = int(row["pred_class_id"])
                gt_class_id = int(row["gt_class_id"])
                if self.detection_require_class_match and pred_class_id != gt_class_id:
                    continue
                if not 1 <= gt_class_id <= 6:
                    continue
                bbox = np.asarray(row["pred_bbox_yxyx"], dtype=np.float64).reshape(-1)
                if bbox.shape != (4,) or not np.isfinite(bbox).all():
                    raise ValueError(
                        f"Invalid pred_bbox_yxyx at manifest row {line_number}: {bbox}"
                    )

                normalized = dict(row)
                normalized["manifest_line"] = line_number
                normalized["pred_index"] = int(row["pred_index"])
                normalized["gt_index"] = int(row["gt_index"])
                normalized["pred_class_id"] = pred_class_id
                normalized["gt_class_id"] = gt_class_id
                normalized["bbox_iou"] = float(row["bbox_iou"])
                normalized["pred_bbox_yxyx"] = bbox.astype(np.float32)
                rows.append(normalized)

        if not rows:
            filters = (
                f"split={self.detection_split!r}, min_iou={self.detection_min_iou}, "
                f"require_class_match={self.detection_require_class_match}"
            )
            raise ValueError(
                f"Detection manifest has no usable rows after filtering ({filters}): "
                f"{self.detection_manifest_path}"
            )
        return rows

    def detection_class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for row in self.detection_rows:
            class_id = int(row["gt_class_id"])
            counts[class_id] = counts.get(class_id, 0) + 1
        return counts

    def clone_for_detection_split(self, split: str) -> "Real275Dataset":
        """Create a lightweight, model-sharing view of another manifest split."""
        if self.detection_manifest_path is None:
            raise ValueError("A detection manifest is required to clone a split")
        cloned = copy.copy(self)
        cloned.detection_split = str(split)
        cloned.detection_rows = cloned._load_detection_manifest()
        cloned.uses_detection_manifest = True
        cloned.ordered_detection_indices = list(range(len(cloned.detection_rows)))
        cloned.dataset_length = len(cloned.detection_rows)
        cloned.image_shuffle = False
        cloned.augmentation = dict(cloned.augmentation)
        cloned.augmentation["enable"] = False
        cloned.augmentation_enable = False
        cloned.set_epoch(0)
        return cloned

    def _sample_rng(self, sample_path: str, obj_idx: int) -> np.random.Generator:
        token = f"{self.seed or 0}|{self.epoch}|{sample_path}|{obj_idx}"
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], byteorder="little", signed=False))

    def _augment_bbox(
            self,
            bbox_xyxy: Tuple[int, int, int, int],
            image_shape: Tuple[int, int],
            rng: np.random.Generator,
    ) -> Tuple[Tuple[int, int, int, int], bool]:
        if not self.augmentation_enable:
            return bbox_xyxy, False
        probability = float(self.augmentation.get("bbox_jitter_prob", 0.0))
        if rng.random() >= probability:
            return bbox_xyxy, False

        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        shift_ratio = abs(float(self.augmentation.get("bbox_shift_ratio", 0.05)))
        scale_range = self.augmentation.get("bbox_scale_range", [0.9, 1.1])
        scale_low, scale_high = sorted(float(value) for value in scale_range)
        scale = float(rng.uniform(scale_low, scale_high))
        center_x = (x1 + x2) * 0.5 + float(rng.uniform(-shift_ratio, shift_ratio)) * width
        center_y = (y1 + y2) * 0.5 + float(rng.uniform(-shift_ratio, shift_ratio)) * height
        half_width = width * scale * 0.5
        half_height = height * scale * 0.5
        image_height, image_width = image_shape[:2]
        jittered = (
            int(np.clip(np.floor(center_x - half_width), 0, image_width - 2)),
            int(np.clip(np.floor(center_y - half_height), 0, image_height - 2)),
            int(np.clip(np.ceil(center_x + half_width), 1, image_width - 1)),
            int(np.clip(np.ceil(center_y + half_height), 1, image_height - 1)),
        )
        if jittered[2] <= jittered[0] or jittered[3] <= jittered[1]:
            return bbox_xyxy, False
        return jittered, True

    def _augment_depth_crop(
            self,
            depth_crop: np.ndarray,
            mask_crop: np.ndarray,
            rng: np.random.Generator,
    ) -> Tuple[np.ndarray, bool]:
        if not self.augmentation_enable:
            return depth_crop, False
        probability = float(self.augmentation.get("depth_noise_prob", 0.0))
        if rng.random() >= probability:
            return depth_crop, False
        return augment_depth_consistently(
            depth_crop,
            mask_crop,
            rng,
            scale_range=self.augmentation.get("depth_scale_range", [1.0, 1.0]),
            bias_m=float(self.augmentation.get("depth_bias_m", 0.0)),
            gaussian_std_m=float(self.augmentation.get("depth_gaussian_std_m", 0.0)),
        ), True

    def _augment_rgb_tensor(self, rgb_resized: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        probability = float(self.augmentation.get("color_jitter_prob", 0.0))
        if self.augmentation_enable and rng.random() < probability:
            rgb_resized = np.asarray(self.color_jitter(Image.fromarray(rgb_resized.astype(np.uint8))))
        return self.transform(rgb_resized)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        sample_paths = list(self.sample_paths)
        detection_indices = list(range(len(self.detection_rows)))
        if self.image_shuffle:
            seed = self.seed if self.seed is not None else 0
            random.Random(seed + epoch).shuffle(sample_paths)
            random.Random(seed + epoch).shuffle(detection_indices)
        self.ordered_sample_paths = sample_paths
        self.ordered_detection_indices = detection_indices

    def _resolve_dataset_length(self, batch_size: Optional[int], num_mini_batch: Optional[int]) -> int:
        if batch_size is None or num_mini_batch is None:
            return len(self.detection_rows) if self.detection_rows else len(self.sample_paths)
        return int(batch_size) * int(num_mini_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Dict:
        if self.uses_detection_manifest:
            row_index = int(index[-1] if isinstance(index, tuple) else index)
            row_index %= len(self.detection_rows)
            candidates = [row_index]
            if self.skip_bad_samples:
                candidates.extend(self._fallback_detection_candidates(row_index))
        elif isinstance(index, tuple):
            sample_path, obj_idx = index
            candidates = [(sample_path, obj_idx)]
            if self.skip_bad_samples:
                candidates.extend(self._fallback_candidates(sample_path))
        else:
            sample_path = self.ordered_sample_paths[index % len(self.ordered_sample_paths)]
            candidates = [(sample_path, 0)]
            if self.skip_bad_samples:
                candidates.extend(self._fallback_candidates(sample_path))

        skipped_errors = []
        for candidate in candidates:
            detection_row = None
            detection_row_index = None
            try:
                if self.uses_detection_manifest:
                    detection_row_index = int(candidate)
                    detection_row = self.detection_rows[detection_row_index]
                    sample_path = self._detection_sample_path(detection_row)
                    obj_idx = int(detection_row["gt_index"])
                    rgb, depth, gts, depth_path = self._load_detection_scene(sample_path)
                    result_path = os.path.join(
                        self.detection_results_dir,
                        *str(detection_row["result_path"]).replace("\\", "/").split("/"),
                    )
                    with open(result_path, "rb") as stream:
                        result_data = load_pickle_ignoring_numpy_align_warning(stream)
                    pred_bboxes = np.asarray(
                        result_data.get("pred_bboxes", []), dtype=np.float32
                    ).reshape(-1, 4)
                    pred_class_ids = np.asarray(
                        result_data.get("pred_class_ids", []), dtype=np.int64
                    ).reshape(-1)
                    pred_index = int(detection_row["pred_index"])
                    if not 0 <= pred_index < len(pred_bboxes):
                        raise IndexError(
                            f"pred_index={pred_index} is outside {result_path} "
                            f"with {len(pred_bboxes)} predictions"
                        )
                    if pred_index >= len(pred_class_ids):
                        raise IndexError(f"Missing pred_class_id[{pred_index}] in {result_path}")
                    if int(pred_class_ids[pred_index]) != int(detection_row["pred_class_id"]):
                        raise ValueError(
                            f"Manifest/result class mismatch at {result_path} pred={pred_index}: "
                            f"{detection_row['pred_class_id']} != {pred_class_ids[pred_index]}"
                        )
                    detection_bbox_yxyx = pred_bboxes[pred_index]
                    if not np.allclose(
                        detection_bbox_yxyx,
                        detection_row["pred_bbox_yxyx"],
                        atol=1.0e-4,
                    ):
                        raise ValueError(
                            f"Manifest/result bbox mismatch at {result_path} pred={pred_index}"
                        )
                    if not 0 <= obj_idx < len(gts.get("bboxes", [])):
                        raise IndexError(
                            f"gt_index={obj_idx} is outside {sample_path}_label.pkl"
                        )
                    if int(gts["class_ids"][obj_idx]) != int(detection_row["gt_class_id"]):
                        raise ValueError(
                            f"Manifest/GT class mismatch at {sample_path} gt={obj_idx}: "
                            f"{detection_row['gt_class_id']} != {gts['class_ids'][obj_idx]}"
                        )
                    pred_masks = load_real275_result_pred_masks(
                        result_data=result_data,
                        result_path=result_path,
                        sample_path=sample_path,
                        image_shape=rgb.shape[:2],
                        instance_count=len(pred_bboxes),
                    )
                    if pred_index >= pred_masks.shape[-1]:
                        raise IndexError(f"Missing pred_mask[{pred_index}] in {result_path}")
                    mask = pred_masks[:, :, pred_index]
                else:
                    sample_path, obj_idx = candidate
                    rgb, depth, mask, gts, depth_path = self._load_scene(sample_path)
                    result_path = None
                    pred_index = None
                    detection_bbox_yxyx = None
            except (
                FileNotFoundError,
                OSError,
                pickle.UnpicklingError,
                KeyError,
                IndexError,
                ValueError,
            ) as exc:
                skipped_errors.append(str(exc))
                if self.skip_bad_samples:
                    continue
                raise

            metric_depth = None
            metric_depth_path = None
            if self.metric_depth_enable:
                if self.metric_depth_source == "same":
                    metric_depth = depth
                    metric_depth_path = depth_path
                else:
                    metric_depth, metric_depth_path = load_depth_metres(
                    sample_path,
                    source=self.metric_depth_source,
                    png_scale=self.metric_depth_png_scale,
                    required=self.metric_depth_required,
                )

            num_instances = len(gts["bboxes"])
            if num_instances == 0:
                continue

            if obj_idx >= num_instances:
                continue

            bbox_yxyx = (
                detection_bbox_yxyx
                if detection_row is not None
                else gts["bboxes"][obj_idx]
            )
            bbox_xyxy = yxyx_to_xyxy(bbox_yxyx, rgb.shape[:2])
            sample_rng = self._sample_rng(
                sample_path,
                detection_row_index if detection_row_index is not None else obj_idx,
            )
            bbox_xyxy, bbox_was_augmented = self._augment_bbox(
                bbox_xyxy,
                rgb.shape[:2],
                sample_rng,
            )
            bbox_yxyx_used = np.asarray(
                [bbox_xyxy[1], bbox_xyxy[0], bbox_xyxy[3], bbox_xyxy[2]],
                dtype=np.int32,
            )
            if detection_row is not None:
                object_mask = self._limit_object_mask_to_bbox(mask, bbox_xyxy)
            else:
                object_mask = self._select_object_mask(
                    mask,
                    gts,
                    obj_idx,
                    bbox_xyxy=bbox_xyxy,
                )
            object_mask, scene_mask_fallback_used = self._ensure_scene_object_mask(
                object_mask,
                bbox_xyxy,
            )
            cropped = crop_scene_fields(
                rgb=rgb,
                depth=depth,
                mask=object_mask,
                bbox_xyxy=bbox_xyxy,
            )

            mask_crop = cropped["mask_crop"].astype(np.uint8)
            mask_center_uv, mask_center_valid, mask_area_ratio = mask_center_from_crop(
                mask_crop,
                bbox_xyxy,
            )
            rgb_crop = cropped["rgb_crop"].copy()
            depth_crop = cropped["depth_crop"].copy()
            if self.depth_mode == "rgb_only":
                depth_was_augmented = False
                point_cloud_crop = np.zeros((*depth_crop.shape, 3), dtype=np.float32)
                point_branch = build_rgb_only_point_branch(self.sample_num)
            else:
                depth_crop, depth_was_augmented = self._augment_depth_crop(
                    depth_crop,
                    mask_crop,
                    sample_rng,
                )
                point_cloud_crop = depth_to_point_cloud(
                    depth_crop,
                    self.cam_k,
                    bbox_xyxy=bbox_xyxy,
                    projection_mode=self.projection_mode,
                )
                point_cloud_crop[mask_crop == 0] = 0
                point_branch = self._load_or_build_point_branch(
                    sample_path=sample_path,
                    obj_idx=obj_idx,
                    bbox_yxyx=bbox_yxyx_used,
                    point_cloud_crop=point_cloud_crop,
                    rgb_crop=rgb_crop,
                    mask_crop=mask_crop,
                    depth_path=depth_path,
                    use_cache=(
                        detection_row is None
                        and not (bbox_was_augmented or depth_was_augmented)
                    ),
                )

            aligned = resize_aligned_modalities(
                rgb_crop=rgb_crop,
                depth_crop=depth_crop,
                mask_crop=mask_crop,
                img_size=self.img_size,
                depth_feature_encoding=self.depth_feature_encoding,
                depth_feature_min_m=self.depth_feature_min_m,
                depth_feature_max_m=self.depth_feature_max_m,
            )

            if len(point_branch["pts_sampled"]) == 0:
                continue
            if self.depth_mode == "rgb_only":
                normals = np.zeros_like(point_branch["pts_sampled"], dtype=np.float32)
            else:
                normals = self._load_or_build_normal_branch(
                    point_branch=point_branch,
                    sample_path=sample_path,
                    obj_idx=obj_idx,
                    bbox_yxyx=bbox_yxyx_used,
                    depth_path=depth_path,
                    use_cache=(
                        detection_row is None
                        and not (bbox_was_augmented or depth_was_augmented)
                    ),
                )

            metric_depth_resized = np.zeros_like(aligned["depth_resized"], dtype=np.float32)
            metric_pts = np.zeros_like(point_branch["pts_sampled"], dtype=np.float32)
            metric_depth_valid = np.zeros((len(point_branch["pts_sampled"]),), dtype=bool)
            metric_depth_crop = None
            if metric_depth is not None:
                if self.metric_depth_source == "same":
                    metric_depth_crop = depth_crop.copy()
                else:
                    metric_cropped = crop_scene_fields(
                        rgb=rgb,
                        depth=metric_depth,
                        mask=object_mask,
                        bbox_xyxy=bbox_xyxy,
                    )
                    metric_depth_crop = metric_cropped["depth_crop"].copy()
                metric_aligned = resize_aligned_modalities(
                    rgb_crop=rgb_crop,
                    depth_crop=metric_depth_crop,
                    mask_crop=mask_crop,
                    img_size=self.img_size,
                    depth_feature_encoding="raw",
                )
                metric_depth_resized = metric_aligned["depth_metric_resized"].astype(np.float32)
                metric_point_cloud_crop = depth_to_point_cloud(
                    metric_depth_crop,
                    self.cam_k,
                    bbox_xyxy=bbox_xyxy,
                    projection_mode=self.projection_mode,
                )
                metric_point_cloud_crop[mask_crop == 0] = 0
                metric_pts, metric_depth_valid = sample_point_map_at_pixels(
                    metric_point_cloud_crop,
                    point_branch["sampled_pixel_indices"],
                )
            pts_local = point_branch["pts_sampled"].astype(np.float32)
            pts_metric = metric_pts.astype(np.float32)
            pts_metric_valid = metric_depth_valid.astype(bool)
            if not np.any(pts_metric_valid):
                pts_metric = pts_local.copy()
                pts_metric_valid = np.isfinite(pts_metric).all(axis=1) & (pts_metric[:, 2] > 0.0)

            rgb_tensor = self._augment_rgb_tensor(
                np.array(aligned["rgb_resized"]),
                sample_rng,
            )

            class_id = int(gts["class_ids"][obj_idx])
            category_label = class_id - 1

            sample = {
                "sample_path": sample_path,
                "object_index": obj_idx,
                "source": self.source_name,
                "source_id": np.array(self.source_id, dtype=np.int64),
                # Selected object bbox in the same format as NOCS labels: [y1, x1, y2, x2].
                "bbox": bbox_yxyx_used,
                "bbox_format": "yxyx",
                "mask_center_uv": mask_center_uv,
                "mask_center_valid": np.asarray(mask_center_valid, dtype=bool),
                "mask_area_ratio": np.asarray(mask_area_ratio, dtype=np.float32),
                "cam_k": np.asarray(self.cam_k, dtype=np.float32),
                "image_hw": np.asarray(rgb.shape[:2], dtype=np.float32),
                # Resized RGB patch aligned with ASD2-Pose naming.
                "rgb": rgb_tensor,
                # Full scene inputs for RGB-only monocular metric-depth models.
                "scene_rgb": self.transform(rgb),
                "scene_object_mask": object_mask.astype(bool),
                "scene_mask_fallback_used": np.asarray(
                    scene_mask_fallback_used, dtype=bool
                ),
                # Dense cropped RGB patch before resize, kept for debugging/visualization.
                "rgb_crop": rgb_crop,
                # Fixed-encoding dense feature derived from the selected geometry depth.
                "depth": aligned["depth_resized"],
                "depth_crop": depth_crop,
                # Raw metric depth in metres, kept separate from the dense encoded feature.
                "metric_depth": metric_depth_resized,
                "metric_depth_crop": metric_depth_crop,
                "metric_pts": metric_pts,
                "metric_depth_valid": metric_depth_valid,
                "pts_local": pts_local,
                "pts_metric": pts_metric,
                "pts_metric_valid": pts_metric_valid,
                # Fixed-length point cloud sampled by FPS from the filtered crop point cloud.
                "pts": pts_local,
                # Filtered crop point cloud before FPS, kept for visualization/debugging.
                "point_cloud_filtered": point_branch["point_cloud_filtered"],
                # Dense point cloud map inside the bbox, kept for visualization.
                "point_cloud_crop": point_cloud_crop,
                # Cropped SAM foreground mask inside the selected bbox.
                "mask_crop": mask_crop,
                # Pixel indices in the original crop for the sampled FPS points.
                "choose": point_branch["sampled_pixel_indices"],
                # RGB values aligned with the sampled point cloud.
                "rgb_points": point_branch["rgb_points_sampled"],
                # Surface normals aligned with the sampled point cloud.
                "normals": normals,
                # RGB values aligned with the filtered crop point cloud before FPS.
                "rgb_points_filtered": point_branch["rgb_points_filtered"],
                "depth_source": self.depth_mode,
                "depth_path": depth_path,
                "metric_depth_source": self.metric_depth_source,
                "metric_depth_path": metric_depth_path,
                "projection_mode": self.projection_mode,
            }
            if metric_depth is not None:
                sample["scene_metric_depth"] = metric_depth.astype(
                    np.float32,
                    copy=False,
                )
            if detection_row is not None:
                sample.update(
                    {
                        "detection_result_path": result_path,
                        "detection_manifest_line": int(detection_row["manifest_line"]),
                        "detection_index": int(pred_index),
                        "detection_pred_class_id": np.asarray(
                            detection_row["pred_class_id"], dtype=np.int64
                        ),
                        "detection_score": np.asarray(
                            detection_row.get("pred_score", -1.0), dtype=np.float32
                        ),
                        "detection_bbox_iou": np.asarray(
                            detection_row["bbox_iou"], dtype=np.float32
                        ),
                    }
                )

            model_key = gts["model_list"][obj_idx]
            if model_key not in self.models:
                raise KeyError(f"Model {model_key} not found in loaded obj_models")

            # Category id of the selected object in the current scene.
            sample["class_id"] = class_id
            # Instance id of the selected object in the current scene.
            sample["instance_id"] = int(gts["instance_ids"][obj_idx])
            # Model identifier used to retrieve the canonical object point cloud.
            sample["model_name"] = model_key
            shape_code, shape_code_valid = self._shape_code_for_model(model_key)
            sample["shape_code"] = shape_code
            sample["shape_code_valid"] = np.array(shape_code_valid, dtype=bool)
            # Zero-indexed category label aligned with ASD2-Pose.
            sample["category_label"] = np.array([category_label], dtype=np.int64)
            detector_class_id = (
                int(detection_row["pred_class_id"])
                if detection_row is not None
                else class_id
            )
            sample["detector_category_label"] = np.array(
                [detector_class_id - 1], dtype=np.int64
            )
            sample["detector_score"] = np.asarray(
                detection_row.get("pred_score", 1.0)
                if detection_row is not None
                else 1.0,
                dtype=np.float32,
            )
            rotation = np.asarray(gts["rotations"][obj_idx], dtype=np.float32)
            if self.canonicalize_symmetry_rotation and category_label in self.sym_ids:
                theta_x = float(rotation[0, 0] + rotation[2, 2])
                theta_y = float(rotation[0, 2] - rotation[2, 0])
                r_norm = math.sqrt(theta_x ** 2 + theta_y ** 2)
                if r_norm > 1.0e-8:
                    s_map = np.array([
                        [theta_x / r_norm, 0.0, -theta_y / r_norm],
                        [0.0, 1.0, 0.0],
                        [theta_y / r_norm, 0.0, theta_x / r_norm],
                    ], dtype=np.float32)
                    rotation = rotation @ s_map
            translation = np.asarray(gts["translations"][obj_idx], dtype=np.float32)
            size_label = (
                np.asarray(gts["scales"][obj_idx], dtype=np.float32)
                * np.asarray(gts["sizes"][obj_idx], dtype=np.float32)
            )
            # Ground-truth rotation matrix of the selected object.
            sample["rotation_label"] = rotation
            # Ground-truth translation vector of the selected object.
            sample["translation_label"] = translation
            # Ground-truth object size, computed as scale * canonical size.
            sample["size_label"] = size_label
            # Canonical model point cloud loaded from obj_models/real_train.pkl.
            sample["model"] = np.asarray(self.models[model_key], dtype=np.float32)
            sample["qo"] = (
                (sample["pts"] - translation[np.newaxis, :])
                / (np.linalg.norm(size_label) + 1.0e-8)
                @ rotation
            ).astype(np.float32)
            # Keep scene-level metadata that is not duplicated by the single-object labels above.
            if "model_list" in gts:
                sample["scene_model_list"] = gts["model_list"]

            return sample

        if skipped_errors:
            joined_errors = " | ".join(skipped_errors[:3])
            raise FileNotFoundError(
                "Failed to load a valid fallback sample after skipping bad sample(s): {}".format(joined_errors)
            )
        raise ValueError("No valid sample with non-overlapping bbox found in the dataset.")

    def _detection_sample_path(self, row: Dict) -> str:
        sample_rel = str(row["sample_rel"]).replace("\\", "/").lstrip("./")
        if sample_rel.lower().startswith("real/"):
            sample_rel = sample_rel[len("Real/") :]
        return os.path.join(self.base_dir, *sample_rel.split("/"))

    def _load_detection_scene(
        self, sample_path: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict, str]:
        rgb_bgr = cv2.imread(sample_path + "_color.png", cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {sample_path}_color.png")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        if self.depth_mode == "rgb_only":
            depth = np.zeros(rgb.shape[:2], dtype=np.float32)
            depth_path = "rgb_only"
        else:
            depth, depth_path = load_depth_metres(
                sample_path,
                source=self.depth_mode,
                png_scale=self.metric_depth_png_scale,
                required=True,
            )
        with open(sample_path + "_label.pkl", "rb") as stream:
            gts = load_pickle_ignoring_numpy_align_warning(stream)
        return rgb, depth.astype(np.float32), gts, str(depth_path)

    @staticmethod
    def _limit_object_mask_to_bbox(
        object_mask: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
    ) -> np.ndarray:
        foreground = mask_to_foreground(object_mask)
        height, width = foreground.shape
        x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
        x1 = int(np.clip(x1, 0, max(width - 1, 0)))
        y1 = int(np.clip(y1, 0, max(height - 1, 0)))
        x2 = int(np.clip(x2, x1 + 1, width))
        y2 = int(np.clip(y2, y1 + 1, height))
        selected = np.zeros_like(foreground, dtype=np.uint8)
        selected[y1:y2, x1:x2] = foreground[y1:y2, x1:x2]
        return selected

    def _fallback_detection_candidates(self, row_index: int) -> List[int]:
        if not self.ordered_detection_indices:
            return []
        try:
            start_index = self.ordered_detection_indices.index(row_index)
        except ValueError:
            start_index = 0
        candidates = []
        max_candidates = min(self.bad_sample_retry, len(self.ordered_detection_indices))
        for offset in range(1, max_candidates):
            candidates.append(
                self.ordered_detection_indices[
                    (start_index + offset) % len(self.ordered_detection_indices)
                ]
            )
        return candidates

    def _load_scene(self, sample_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, str]:
        rgb_bgr = cv2.imread(sample_path + "_color.png", cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {sample_path}_color.png")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        if self.depth_mode == "rgb_only":
            depth = np.zeros(rgb.shape[:2], dtype=np.float32)
            depth_path = "rgb_only"
        else:
            depth, depth_path = load_depth_metres(
                sample_path,
                source=self.depth_mode,
                png_scale=self.metric_depth_png_scale,
                required=True,
            )

        if self.mask_mode == "sam":
            mask_path = sample_path + "_mask_sam.png"
        elif self.mask_mode == "instance":
            mask_path = sample_path + "_mask.png"
        else:
            raise ValueError(f"Unsupported mask_mode={self.mask_mode}")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        with open(sample_path + "_label.pkl", "rb") as f:
            gts = pickle.load(f)

        return rgb, depth.astype(np.float32), mask, gts, str(depth_path)

    def _select_object_mask(
        self,
        mask: np.ndarray,
        gts: Dict,
        obj_idx: int,
        bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        if self.mask_mode == "sam":
            foreground = mask_to_foreground(mask)
            if bbox_xyxy is None:
                bbox_xyxy = yxyx_to_xyxy(gts["bboxes"][obj_idx], foreground.shape)
            height, width = foreground.shape
            x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
            x1 = int(np.clip(x1, 0, max(width - 1, 0)))
            y1 = int(np.clip(y1, 0, max(height - 1, 0)))
            x2 = int(np.clip(x2, x1 + 1, width))
            y2 = int(np.clip(y2, y1 + 1, height))
            selected = np.zeros_like(foreground, dtype=np.uint8)
            selected[y1:y2, x1:x2] = foreground[y1:y2, x1:x2]
            return selected
        if self.mask_mode != "instance":
            raise ValueError(f"Unsupported mask_mode={self.mask_mode}")
        if "instance_ids" not in gts:
            raise KeyError("instance_ids missing from instance-mask label data")
        instance_id = int(gts["instance_ids"][obj_idx])
        instance_mask = mask[:, :, 2] if mask.ndim == 3 else mask
        selected = np.zeros(instance_mask.shape, dtype=np.uint8)
        selected[instance_mask == instance_id] = 1
        return selected

    def _ensure_scene_object_mask(
        self,
        object_mask: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, bool]:
        object_mask = mask_to_foreground(object_mask)
        if (
            not self.scene_mask_fallback_to_bbox
            or np.count_nonzero(object_mask) >= self.scene_mask_min_pixels
        ):
            return object_mask, False

        height, width = object_mask.shape
        x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
        x1 = int(np.clip(x1, 0, max(width - 1, 0)))
        y1 = int(np.clip(y1, 0, max(height - 1, 0)))
        x2 = int(np.clip(x2, x1 + 1, width))
        y2 = int(np.clip(y2, y1 + 1, height))
        fallback = np.zeros((height, width), dtype=np.uint8)
        fallback[y1:y2, x1:x2] = 1
        return fallback, True

    def _shape_code_for_model(self, model_key: str) -> Tuple[torch.Tensor, bool]:
        for latent_key in _model_name_candidates(model_key):
            latent_key = latent_key.replace("_norm", "")
            if latent_key in self.latent_codes:
                return self.latent_codes[latent_key].clone(), True
        if self.shape_code_required:
            raise KeyError(f"Latent code for model {model_key} not found in {self.latent_code_path}")
        return torch.zeros((self.shape_code_dim,), dtype=torch.float32), False

    def _fallback_candidates(self, sample_path: str) -> List[Tuple[str, int]]:
        try:
            start_idx = self.ordered_sample_paths.index(sample_path)
        except ValueError:
            start_idx = 0

        candidates = []
        total_paths = len(self.ordered_sample_paths)
        for offset in range(1, min(self.bad_sample_retry, total_paths - 1) + 1):
            fallback_path = self.ordered_sample_paths[(start_idx + offset) % total_paths]
            label_path = fallback_path + "_label.pkl"
            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, "rb") as f:
                    gts = pickle.load(f)
            except (OSError, pickle.UnpicklingError):
                continue
            valid_indices = find_non_overlapping_indices(gts.get("bboxes", []), self.image_shape)
            candidates.extend((fallback_path, obj_idx) for obj_idx in valid_indices)
            if candidates:
                break
        return candidates

    def _load_or_build_point_branch(
            self,
            sample_path: str,
            obj_idx: int,
            bbox_yxyx,
            point_cloud_crop: np.ndarray,
            rgb_crop: np.ndarray,
            mask_crop: np.ndarray,
            depth_path: str,
            use_cache: bool,
    ) -> Dict[str, np.ndarray]:
        cache_path = None
        cache_enabled = self.denoise_cache_enable and bool(use_cache)
        if cache_enabled:
            mask_suffix = "_mask_sam.png" if self.mask_mode == "sam" else "_mask.png"
            cache_path = real275_denoise_cache_path(
                sample_path=sample_path,
                obj_idx=obj_idx,
                bbox_yxyx=bbox_yxyx,
                data_dir=self.data_dir,
                sample_num=self.sample_num,
                cache_dir=self.denoise_cache_dir,
                depth_path=depth_path,
                mask_path=sample_path + mask_suffix,
                projection_mode=self.projection_mode,
            )
            if self.denoise_cache_mode != "rebuild" and os.path.exists(cache_path):
                return load_point_branch_cache(cache_path)
            if self.denoise_cache_mode == "read_only":
                raise FileNotFoundError(
                    f"Missing denoise cache for {sample_path} obj={obj_idx}: {cache_path}. "
                    "Run data_preprocess/build_real275_denoise_cache.py first or use read_write mode."
                )

        point_branch = filter_and_sample_points(
            point_cloud_crop=point_cloud_crop,
            rgb_crop=rgb_crop,
            mask_crop=mask_crop,
            sample_num=self.sample_num,
        )
        if (
            cache_enabled
            and cache_path is not None
            and self.denoise_cache_mode in {"read_write", "rebuild"}
            and len(point_branch["pts_sampled"]) > 0
        ):
            save_point_branch_cache(cache_path, point_branch, bbox_yxyx, obj_idx, sample_path)
        return point_branch

    def _load_or_build_normal_branch(
            self,
            point_branch: Dict[str, np.ndarray],
            sample_path: str,
            obj_idx: int,
            bbox_yxyx,
            depth_path: str,
            use_cache: bool,
    ) -> np.ndarray:
        normal_cache_path = None
        cache_enabled = self.denoise_cache_enable and bool(use_cache)
        if cache_enabled:
            mask_suffix = "_mask_sam.png" if self.mask_mode == "sam" else "_mask.png"
            point_cache_path = real275_denoise_cache_path(
                sample_path=sample_path,
                obj_idx=obj_idx,
                bbox_yxyx=bbox_yxyx,
                data_dir=self.data_dir,
                sample_num=self.sample_num,
                cache_dir=self.denoise_cache_dir,
                depth_path=depth_path,
                mask_path=sample_path + mask_suffix,
                projection_mode=self.projection_mode,
            )
            normal_cache_path = real275_normal_cache_path(point_cache_path)
            if self.denoise_cache_mode != "rebuild" and os.path.exists(normal_cache_path):
                normals = load_normal_cache(normal_cache_path)
                if normals.shape == point_branch["pts_sampled"].shape:
                    return normals

        normals = build_normals_for_point_branch(point_branch)
        if (
            cache_enabled
            and normal_cache_path is not None
            and self.denoise_cache_mode != "off"
            and len(normals) > 0
        ):
            save_normal_cache(normal_cache_path, normals)
        return normals


class Real275ImageObjectSampler(Sampler):
    def __init__(self, dataset: Real275Dataset):
        self.dataset = dataset

    def __iter__(self):
        yielded = 0
        total = len(self.dataset)

        while yielded < total:
            yielded_in_pass = 0
            if self.dataset.uses_detection_manifest:
                for row_index in self.dataset.ordered_detection_indices:
                    if yielded >= total:
                        break
                    yielded += 1
                    yielded_in_pass += 1
                    yield row_index
                if yielded_in_pass == 0:
                    raise ValueError("Detection manifest has no usable object samples.")
                continue

            for sample_path in self.dataset.ordered_sample_paths:
                if yielded >= total:
                    break

                label_path = sample_path + "_label.pkl"
                if not os.path.exists(label_path):
                    continue

                with open(label_path, "rb") as f:
                    gts = pickle.load(f)

                bboxes = gts.get("bboxes", [])
                if len(bboxes) == 0:
                    continue

                valid_indices = find_non_overlapping_indices(bboxes, self.dataset.image_shape)
                for obj_idx in valid_indices:
                    if yielded >= total:
                        break
                    yielded += 1
                    yielded_in_pass += 1
                    yield sample_path, obj_idx

            if yielded_in_pass == 0:
                raise ValueError("No valid object samples found in the dataset.")

    def __len__(self) -> int:
        return len(self.dataset)


class ClassBalancedImageObjectSampler(Sampler):
    def __init__(
        self,
        dataset: Real275Dataset,
        class_weights: Optional[Dict] = None,
    ):
        self.dataset = dataset
        self.class_weights = class_weights or {}
        self.class_buckets = self._build_class_buckets()
        if not self.class_buckets:
            raise ValueError("ClassBalancedImageObjectSampler found no valid object samples.")

    def _class_weight(self, class_id: int) -> float:
        class_names = ["bottle", "bowl", "camera", "can", "laptop", "mug"]
        zero_idx = class_id - 1
        value = self.class_weights.get(str(class_id), None)
        if value is None and 0 <= zero_idx < len(class_names):
            value = self.class_weights.get(class_names[zero_idx], None)
        if value is None:
            value = self.class_weights.get(str(zero_idx), 1.0)
        return max(float(value), 0.0)

    def _build_class_buckets(self) -> Dict[int, List]:
        buckets: Dict[int, List] = {}
        if self.dataset.uses_detection_manifest:
            for row_index, row in enumerate(self.dataset.detection_rows):
                class_id = int(row["gt_class_id"])
                buckets.setdefault(class_id, []).append(row_index)
            return buckets

        for sample_path in self.dataset.sample_paths:
            label_path = sample_path + "_label.pkl"
            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, "rb") as f:
                    gts = pickle.load(f)
            except (OSError, pickle.UnpicklingError):
                continue
            bboxes = gts.get("bboxes", [])
            class_ids = gts.get("class_ids", [])
            valid_indices = find_non_overlapping_indices(bboxes, self.dataset.image_shape)
            for obj_idx in valid_indices:
                if obj_idx >= len(class_ids):
                    continue
                class_id = int(class_ids[obj_idx])
                buckets.setdefault(class_id, []).append((sample_path, obj_idx))
        return buckets

    def __iter__(self):
        rng = random.Random((self.dataset.seed or 0) + int(getattr(self.dataset, "epoch", 0)))
        class_ids = sorted(self.class_buckets)
        weights = [self._class_weight(class_id) for class_id in class_ids]
        if sum(weights) <= 0.0:
            weights = [1.0 for _ in class_ids]
        for _ in range(len(self.dataset)):
            class_id = rng.choices(class_ids, weights=weights, k=1)[0]
            yield rng.choice(self.class_buckets[class_id])

    def __len__(self) -> int:
        return len(self.dataset)


class MixedNocsDataset(Dataset):
    def __init__(self, datasets: Dict[str, Real275Dataset]):
        if not datasets:
            raise ValueError("MixedNocsDataset requires at least one source dataset.")
        self.datasets = datasets

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets.values():
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 3:
            raise TypeError("MixedNocsDataset expects sampler indices: (source, sample_path, obj_idx)")
        source, sample_path, obj_idx = index
        if source not in self.datasets:
            raise KeyError(f"Unknown mixed dataset source: {source}")
        return self.datasets[source][(sample_path, obj_idx)]


class MixedImageObjectSampler(Sampler):
    def __init__(
        self,
        real_dataset: Real275Dataset,
        synthetic_dataset: Real275Dataset,
        real_bs: int,
        syn_bs: int,
        num_mini_batch: int,
        class_balanced: bool = False,
        class_weights: Optional[Dict] = None,
        class_balance_power: float = 0.5,
    ):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.real_bs = max(0, int(real_bs))
        self.syn_bs = max(0, int(syn_bs))
        self.num_mini_batch = int(num_mini_batch)
        self.class_balanced = bool(class_balanced)
        self.class_weights = class_weights or {}
        self.class_balance_power = min(1.0, max(0.0, float(class_balance_power)))
        self.balance_strategy = (
            "adaptive_streaming_rejection" if self.class_balanced else "natural_stream"
        )
        if self.real_bs + self.syn_bs <= 0:
            raise ValueError("MixedImageObjectSampler requires real_bs + syn_bs > 0")
        if self.num_mini_batch <= 0:
            raise ValueError("num_mini_batch must be positive")

    def _class_multiplier(self, class_id: int) -> float:
        class_names = ["bottle", "bowl", "camera", "can", "laptop", "mug"]
        zero_idx = class_id - 1
        value = self.class_weights.get(str(class_id), None)
        if value is None and 0 <= zero_idx < len(class_names):
            value = self.class_weights.get(class_names[zero_idx], None)
        if value is None:
            value = self.class_weights.get(str(zero_idx), 1.0)
        return max(float(value), 0.0)

    def _labeled_object_stream(self, dataset: Real275Dataset):
        while True:
            yielded_in_pass = 0
            for sample_path in dataset.ordered_sample_paths:
                label_path = sample_path + "_label.pkl"
                if not os.path.exists(label_path):
                    continue
                try:
                    with open(label_path, "rb") as f:
                        gts = pickle.load(f)
                except (OSError, pickle.UnpicklingError):
                    continue
                bboxes = gts.get("bboxes", [])
                class_ids = gts.get("class_ids", [])
                for obj_idx in find_non_overlapping_indices(
                    bboxes,
                    dataset.image_shape,
                ):
                    if obj_idx >= len(class_ids):
                        continue
                    class_id = int(class_ids[obj_idx])
                    if class_id <= 0:
                        continue
                    yielded_in_pass += 1
                    yield dataset.source_name, sample_path, obj_idx, class_id
            if yielded_in_pass == 0:
                raise ValueError(
                    f"No valid object samples found in dataset source={dataset.source_name}."
                )

    def _balanced_object_stream(self, dataset: Real275Dataset, seed_offset: int):
        """Balance classes online without pre-indexing every label file.

        If the natural class frequency is f(c), accepting class c with a
        probability proportional to f(c)^(-power) produces the requested
        smoothed distribution f(c)^(1-power).  Running counts estimate f(c)
        from the already shuffled stream, so startup reads no label PKLs and
        every epoch can draw from the full source dataset.
        """
        epoch = int(getattr(dataset, "epoch", 0))
        rng = random.Random((dataset.seed or 0) + epoch + int(seed_offset))
        multipliers = {
            class_id: self._class_multiplier(class_id)
            for class_id in range(1, 7)
        }
        max_multiplier = max(multipliers.values(), default=0.0)
        if max_multiplier <= 0.0:
            multipliers = {class_id: 1.0 for class_id in range(1, 7)}
            max_multiplier = 1.0

        observed_counts: Dict[int, int] = {}
        candidate_stream = self._labeled_object_stream(dataset)
        for source, sample_path, obj_idx, class_id in candidate_stream:
            observed_counts[class_id] = observed_counts.get(class_id, 0) + 1
            reference_count = min(observed_counts.values())
            frequency_correction = (
                reference_count / float(observed_counts[class_id])
            ) ** self.class_balance_power
            multiplier = multipliers.get(class_id, 1.0) / max_multiplier
            acceptance_probability = min(
                1.0,
                max(0.0, multiplier * frequency_correction),
            )
            if rng.random() <= acceptance_probability:
                yield source, sample_path, obj_idx

    def _object_stream(self, dataset: Real275Dataset):
        while True:
            yielded_in_pass = 0
            for sample_path in dataset.ordered_sample_paths:
                label_path = sample_path + "_label.pkl"
                if not os.path.exists(label_path):
                    continue
                with open(label_path, "rb") as f:
                    gts = pickle.load(f)
                bboxes = gts.get("bboxes", [])
                if len(bboxes) == 0:
                    continue
                valid_indices = find_non_overlapping_indices(bboxes, dataset.image_shape)
                for obj_idx in valid_indices:
                    yielded_in_pass += 1
                    yield dataset.source_name, sample_path, obj_idx
            if yielded_in_pass == 0:
                raise ValueError(f"No valid object samples found in dataset source={dataset.source_name}.")

    def __iter__(self):
        if self.class_balanced:
            real_iter = self._balanced_object_stream(self.real_dataset, 1009)
            syn_iter = self._balanced_object_stream(self.synthetic_dataset, 2017)
        else:
            real_iter = self._object_stream(self.real_dataset)
            syn_iter = self._object_stream(self.synthetic_dataset)
        for _ in range(self.num_mini_batch):
            for _ in range(self.real_bs):
                yield next(real_iter)
            for _ in range(self.syn_bs):
                yield next(syn_iter)

    def __len__(self) -> int:
        return (self.real_bs + self.syn_bs) * self.num_mini_batch


class Real275TestDataset(Dataset):
    """REAL275 evaluation dataset backed by NOCS result PKLs."""

    def __init__(
            self,
            base_dir: str,
            split_file: str = "test_list_all.txt",
            cam_k: Optional[List[float]] = None,
            seed: Optional[int] = None,
            img_size: int = 224,
            sample_num: int = 2048,
            category_names: Optional[List[str]] = None,
            rgb_enhance_mode: str = "none",
            rgb_brightness: float = 0.0,
            rgb_contrast: float = 1.0,
            rgb_gamma: float = 1.0,
            rgb_clahe_clip_limit: float = 2.0,
            rgb_clahe_tile_grid_size: int = 8,
            rgb_enhance_classes: Optional[List[str]] = None,
            depth_mode: str = "legacy_estimated",
            metric_depth_source: str = "same",
            metric_depth_required: bool = True,
            metric_depth_png_scale: float = 1000.0,
            projection_mode: str = "full_image",
            depth_feature_encoding: str = "raw",
            depth_feature_min_m: float = 0.05,
            depth_feature_max_m: float = 5.0,
            segmentation_results_dir: Optional[str] = None,
    ) -> None:
        self.data_dir = base_dir
        self.base_dir = os.path.join(base_dir, "Real")
        self.split_file = split_file
        self.segmentation_results_dir = (
            os.path.abspath(os.path.expanduser(segmentation_results_dir))
            if segmentation_results_dir
            else os.path.join(base_dir, "segmentation_results", "REAL275")
        )
        self.result_pkl_list = sorted(
            glob.glob(os.path.join(self.segmentation_results_dir, "results_*.pkl"))
        )
        if not self.result_pkl_list:
            raise FileNotFoundError(
                f"No REAL275 segmentation result files found under: "
                f"{self.segmentation_results_dir}"
            )
        self.cam_k = cam_k or [591.0125, 590.16775, 322.525, 244.11084]
        self.rng = random.Random(seed)
        self.gt_root = os.path.join(base_dir,"gts","real_test")
        self.img_size = img_size
        self.sample_num = sample_num
        self.category_names = category_names or ["bottle", "bowl", "camera", "can", "laptop", "mug"]
        self.category_name_to_id = {name: idx for idx, name in enumerate(self.category_names)}
        self.rgb_enhance_mode = str(rgb_enhance_mode or "none")
        if self.rgb_enhance_mode not in VALID_RGB_ENHANCE_MODES:
            raise ValueError(
                f"Unsupported rgb_enhance_mode={self.rgb_enhance_mode}. "
                f"Expected one of {sorted(VALID_RGB_ENHANCE_MODES)}"
            )
        self.rgb_brightness = rgb_brightness
        self.rgb_contrast = rgb_contrast
        self.rgb_gamma = rgb_gamma
        self.rgb_clahe_clip_limit = rgb_clahe_clip_limit
        self.rgb_clahe_tile_grid_size = rgb_clahe_tile_grid_size
        self.rgb_enhance_classes = (
            {str(name) for name in rgb_enhance_classes}
            if rgb_enhance_classes
            else None
        )
        self.depth_mode = str(depth_mode).lower()
        self.metric_depth_source = str(metric_depth_source).lower()
        self.metric_depth_required = bool(metric_depth_required)
        self.metric_depth_png_scale = float(metric_depth_png_scale)
        self.projection_mode = str(projection_mode)
        if self.projection_mode not in VALID_PROJECTION_MODES:
            raise ValueError(
                f"projection_mode must be one of {sorted(VALID_PROJECTION_MODES)}, "
                f"got {self.projection_mode!r}"
            )
        self.depth_feature_encoding = str(depth_feature_encoding)
        self.depth_feature_min_m = float(depth_feature_min_m)
        self.depth_feature_max_m = float(depth_feature_max_m)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.result_pkl_list)

    def _requires_valid_metric_points(self) -> bool:
        # RGB-only models consume the full scene map, while their point samples
        # are compatibility placeholders at pixel (0, 0).
        return self.metric_depth_required and self.depth_mode != "rgb_only"

    @staticmethod
    def _limit_object_mask_to_bbox(
        object_mask: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
    ) -> np.ndarray:
        foreground = mask_to_foreground(object_mask)
        height, width = foreground.shape
        x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
        x1 = int(np.clip(x1, 0, max(width - 1, 0)))
        y1 = int(np.clip(y1, 0, max(height - 1, 0)))
        x2 = int(np.clip(x2, x1 + 1, width))
        y2 = int(np.clip(y2, y1 + 1, height))
        selected = np.zeros_like(foreground, dtype=np.uint8)
        selected[y1:y2, x1:x2] = foreground[y1:y2, x1:x2]
        return selected

    def __getitem__(self, index: int) -> Dict:
        dataset_length = len(self.result_pkl_list)

        for offset in range(dataset_length):
            current_index = (index + offset) % dataset_length
            result_path = self.result_pkl_list[current_index]
            with open(result_path, "rb") as f:
                result_data = load_pickle_ignoring_numpy_align_warning(f)

            image_path = result_data.get("image_path")
            if image_path:
                sample_path = resolve_real275_result_image_path(self.data_dir, image_path)
            else:
                sample_path = sample_path_from_real275_result_name(self.data_dir, result_path)
            rgb_bgr = cv2.imread(sample_path + "_color.png", cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(f"Failed to read RGB image: {sample_path}_color.png")
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            if self.depth_mode == "rgb_only":
                depth = np.zeros(rgb.shape[:2], dtype=np.float32)
                depth_path = "rgb_only"
            else:
                depth, depth_path = load_depth_metres(
                    sample_path,
                    source=self.depth_mode,
                    png_scale=self.metric_depth_png_scale,
                    required=True,
                )
            if self.metric_depth_source == "none":
                metric_depth = None
                metric_depth_path = None
            elif self.metric_depth_source == "same":
                metric_depth = depth
                metric_depth_path = depth_path
            else:
                metric_depth, metric_depth_path = load_depth_metres(
                    sample_path,
                    source=self.metric_depth_source,
                    png_scale=self.metric_depth_png_scale,
                    required=self.metric_depth_required,
                )

            bboxes = result_data.get("pred_bboxes", [])
            pred_class_ids_raw = result_data.get("pred_class_ids", [])
            scores = result_data.get("pred_scores", [])
            if len(bboxes) == 0:
                continue
            pred_masks = load_real275_result_pred_masks(
                result_data=result_data,
                result_path=result_path,
                sample_path=sample_path,
                image_shape=rgb.shape[:2],
                instance_count=len(bboxes),
            )

            rgb_tensors = []
            depth_tensors = []
            pts_tensors = []
            metric_depth_tensors = []
            metric_pts_tensors = []
            metric_valid_tensors = []
            choose_tensors = []
            rgb_point_tensors = []
            mask_center_tensors = []
            mask_center_valid_tensors = []
            mask_area_ratio_tensors = []
            scene_object_masks = []
            pred_class_ids = []
            pred_bboxes = []
            pred_scores = []
            pred_label_names = []
            valid_object_indices = []

            for obj_idx, bbox_yxyx in enumerate(bboxes):
                if obj_idx >= pred_masks.shape[-1]:
                    continue

                class_id = int(pred_class_ids_raw[obj_idx])
                category_label = class_id - 1
                if category_label < 0 or category_label >= len(self.category_names):
                    continue
                label_name = self.category_names[category_label]

                bbox_xyxy = yxyx_to_xyxy(bbox_yxyx, rgb.shape[:2])
                object_mask = self._limit_object_mask_to_bbox(
                    pred_masks[:, :, obj_idx],
                    bbox_xyxy,
                )
                cropped = crop_scene_fields(
                    rgb=rgb,
                    depth=depth,
                    mask=object_mask,
                    bbox_xyxy=bbox_xyxy,
                )
                rgb_crop = cropped["rgb_crop"].copy()
                depth_crop = cropped["depth_crop"].copy()
                mask_crop = cropped["mask_crop"].astype(np.uint8)
                mask_center_uv, mask_center_valid, mask_area_ratio = mask_center_from_crop(
                    mask_crop,
                    bbox_xyxy,
                )
                if (
                        self.rgb_enhance_mode != "none"
                        and (
                            self.rgb_enhance_classes is None
                            or label_name in self.rgb_enhance_classes
                        )
                ):
                    rgb_crop = enhance_rgb_crop(
                        rgb_crop=rgb_crop,
                        mode=self.rgb_enhance_mode,
                        brightness=self.rgb_brightness,
                        contrast=self.rgb_contrast,
                        gamma=self.rgb_gamma,
                        clahe_clip_limit=self.rgb_clahe_clip_limit,
                        clahe_tile_grid_size=self.rgb_clahe_tile_grid_size,
                    )
                if self.depth_mode == "rgb_only":
                    point_cloud_crop = np.zeros((*depth_crop.shape, 3), dtype=np.float32)
                    point_branch = build_rgb_only_point_branch(self.sample_num)
                else:
                    point_cloud_crop = depth_to_point_cloud(
                        depth_crop,
                        self.cam_k,
                        bbox_xyxy=bbox_xyxy,
                        projection_mode=self.projection_mode,
                    )
                    point_cloud_crop[mask_crop == 0] = 0
                    point_branch = filter_and_sample_points(
                        point_cloud_crop=point_cloud_crop,
                        rgb_crop=rgb_crop,
                        mask_crop=mask_crop,
                        sample_num=self.sample_num,
                    )
                if len(point_branch["pts_sampled"]) == 0:
                    continue

                aligned = resize_aligned_modalities(
                    rgb_crop=rgb_crop,
                    depth_crop=depth_crop,
                    mask_crop=mask_crop,
                    img_size=self.img_size,
                    depth_feature_encoding=self.depth_feature_encoding,
                    depth_feature_min_m=self.depth_feature_min_m,
                    depth_feature_max_m=self.depth_feature_max_m,
                )

                metric_depth_resized = np.zeros_like(
                    aligned["depth_metric_resized"],
                    dtype=np.float32,
                )
                metric_pts = np.zeros_like(point_branch["pts_sampled"], dtype=np.float32)
                metric_valid = np.zeros((len(point_branch["pts_sampled"]),), dtype=bool)
                if metric_depth is not None:
                    if self.metric_depth_source == "same":
                        metric_depth_crop = depth_crop
                    else:
                        metric_cropped = crop_scene_fields(
                            rgb=rgb,
                            depth=metric_depth,
                            mask=object_mask,
                            bbox_xyxy=bbox_xyxy,
                        )
                        metric_depth_crop = metric_cropped["depth_crop"].copy()
                    metric_resized, _ = resize_masked_depth(
                        metric_depth_crop,
                        mask_crop,
                        self.img_size,
                    )
                    metric_depth_resized = metric_resized.astype(np.float32)
                    metric_point_cloud_crop = depth_to_point_cloud(
                        metric_depth_crop,
                        self.cam_k,
                        bbox_xyxy=bbox_xyxy,
                        projection_mode=self.projection_mode,
                    )
                    metric_point_cloud_crop[mask_crop == 0] = 0
                    metric_pts, metric_valid = sample_point_map_at_pixels(
                        metric_point_cloud_crop,
                        point_branch["sampled_pixel_indices"],
                    )
                if not np.any(metric_valid):
                    if self._requires_valid_metric_points():
                        continue
                    # Keep the metric branch explicitly invalid. Falling back to
                    # pts_local here would silently relabel crop-local geometry
                    # as metre-space camera coordinates during evaluation.
                    metric_pts.fill(0.0)
                    metric_valid.fill(False)

                rgb_tensors.append(self.transform(np.array(aligned["rgb_resized"])))
                depth_tensors.append(np.asarray(aligned["depth_resized"], dtype=np.float32))
                pts_local = np.asarray(point_branch["pts_sampled"], dtype=np.float32)
                pts_tensors.append(pts_local)
                metric_depth_tensors.append(metric_depth_resized)
                metric_pts_tensors.append(metric_pts.astype(np.float32))
                metric_valid_tensors.append(metric_valid.astype(bool))
                choose_tensors.append(np.asarray(point_branch["sampled_pixel_indices"], dtype=np.int64))
                rgb_point_tensors.append(np.asarray(point_branch["rgb_points_sampled"], dtype=np.float32))
                mask_center_tensors.append(mask_center_uv)
                mask_center_valid_tensors.append(mask_center_valid)
                mask_area_ratio_tensors.append(mask_area_ratio)
                scene_object_masks.append(object_mask.astype(bool))
                pred_class_ids.append(category_label + 1)
                pred_bboxes.append(np.asarray(bbox_yxyx, dtype=np.float32))
                pred_scores.append(float(scores[obj_idx]) if obj_idx < len(scores) else -1.0)
                pred_label_names.append(label_name)
                valid_object_indices.append(obj_idx)

            if not rgb_tensors:
                continue

            sample = {
                "sample_path"          : sample_path,
                "result_path"          : result_path,
                "ori_img"              : rgb,
                "object_index"         : np.asarray(valid_object_indices, dtype=np.int64),
                "bbox"                 : np.stack(pred_bboxes).astype(np.float32),
                "bbox_format"          : "yxyx",
                "rgb"                  : torch.stack(rgb_tensors, dim=0),
                "scene_rgb"            : self.transform(rgb),
                "scene_object_mask"    : np.stack(scene_object_masks).astype(bool),
                "depth"                : np.stack(depth_tensors).astype(np.float32),
                "metric_depth"         : np.stack(metric_depth_tensors).astype(np.float32),
                "pts"                  : np.stack(pts_tensors).astype(np.float32),
                "pts_local"            : np.stack(pts_tensors).astype(np.float32),
                "pts_metric"           : np.stack(metric_pts_tensors).astype(np.float32),
                "pts_metric_valid"     : np.stack(metric_valid_tensors).astype(bool),
                "choose"               : np.stack(choose_tensors).astype(np.int64),
                "rgb_points"           : np.stack(rgb_point_tensors).astype(np.float32),
                "category_label"       : (np.asarray(pred_class_ids, dtype=np.int64) - 1).reshape(-1, 1),
                "label_name"           : pred_label_names,
                "score"                : np.asarray(pred_scores, dtype=np.float32),
                "gt_class_ids"         : np.asarray(result_data["gt_class_ids"], dtype=np.int64),
                "gt_bboxes"            : np.asarray(result_data["gt_bboxes"], dtype=np.float32),
                "gt_RTs"               : np.asarray(result_data["gt_RTs"], dtype=np.float32),
                "gt_scales"            : np.asarray(result_data["gt_scales"], dtype=np.float32),
                "gt_handle_visibility" : np.asarray(result_data["gt_handle_visibility"], dtype=np.float32),
                "pred_class_ids"       : np.asarray(pred_class_ids, dtype=np.int64),
                "pred_bboxes"          : np.stack(pred_bboxes).astype(np.float32),
                "pred_scores"          : np.asarray(pred_scores, dtype=np.float32),
                "mask_center_uv"       : np.stack(mask_center_tensors).astype(np.float32),
                "mask_center_valid"    : np.asarray(mask_center_valid_tensors, dtype=bool),
                "mask_area_ratio"      : np.asarray(mask_area_ratio_tensors, dtype=np.float32),
                "cam_k"                : np.repeat(
                    np.asarray(self.cam_k, dtype=np.float32)[None, :],
                    len(pred_bboxes),
                    axis=0,
                ),
                "image_hw"             : np.repeat(
                    np.asarray(rgb.shape[:2], dtype=np.float32)[None, :],
                    len(pred_bboxes),
                    axis=0,
                ),
                "gt_image_path"        : result_data.get("image_path", sample_path),
                "depth_source"         : self.depth_mode,
                "depth_path"           : depth_path,
                "metric_depth_source"  : self.metric_depth_source,
                "metric_depth_path"    : metric_depth_path,
                "projection_mode"      : self.projection_mode,
            }
            if metric_depth is not None:
                sample["scene_metric_depth"] = metric_depth.astype(
                    np.float32,
                    copy=False,
                )

            return sample

        raise ValueError("No valid test sample with predicted instances found in the dataset.")
