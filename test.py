import argparse
import glob
import logging
import os
import pickle
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data_provider.nocs_dataset import Real275TestDataset, test_collate_fn
from utils.checkpoint_utils import load_checkpoint
from utils.config_utils import load_config
from utils.evaluation_utils import (
    DEFAULT_EVAL_PROTOCOL,
    EVAL_PROTOCOLS,
    evaluate,
    normalize_eval_protocol,
)
from utils.logger_utils import get_logger
from utils.runtime_utils import count_parameters, set_cuda_visible_devices
from utils.solver import TestingSolver
from tools.notify_tool import notify_test_end, notify_test_start


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "data_provider"))
sys.path.append(os.path.join(BASE_DIR, "model"))
sys.path.append(os.path.join(BASE_DIR, "utils"))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate GeoQueryPose on REAL275"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU id, consistent with train.py",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="test DataLoader worker count; defaults to test.num_workers or 10",
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=0,
        help="limit inference to the first N test images; 0 uses the full split",
    )
    parser.add_argument(
        "--test-image-offset",
        type=int,
        default=0,
        help="start at this test-image index; result filenames retain the original index",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/duo_sia_pose.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint file to evaluate; defaults to checkpoint in the config",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="none",
        help="name appended to the output directory",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="directory to save test outputs",
    )
    parser.add_argument(
        "--only-eval",
        action="store_true",
        default=False,
        help="skip inference and evaluate existing result pkl files in save path",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        default=False,
        help="run inference and save result PKLs without the dense built-in evaluation",
    )
    parser.add_argument(
        "--capture-direct-rs",
        action="store_true",
        default=False,
        help="capture deterministic R/S heads only; requires --inference-only",
    )
    parser.add_argument(
        "--rotation-output-mode",
        choices=[
            "diffusion",
            "direct",
            "structured",
            "v6",
            "correspondence_v9",
        ],
        default=None,
        help="override test.rotation_output_mode without retraining",
    )
    parser.add_argument(
        "--size-output-mode",
        choices=["blend", "diffusion", "direct", "structured"],
        default=None,
        help="override test.size_output_mode without retraining",
    )
    parser.add_argument(
        "--direct-size-blend",
        type=float,
        default=None,
        help=(
            "override diffusion.direct_size_blend for blend size output; "
            "0 uses diffusion S and 1 uses direct S"
        ),
    )
    parser.add_argument(
        "--sanity-gt",
        action="store_true",
        default=False,
        help="write pred=gt result files to validate the evaluation pipeline",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        default=False,
        help="only compute numeric diagnostics from existing results_*.pkl files",
    )
    parser.add_argument(
        "--denoise-sanity",
        action="store_true",
        default=False,
        help="run one-step GT-conditioned denoise diagnostics instead of DDIM inference",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        default=None,
        choices=["raw", "norm"],
        help="how to convert model size output into pred_RTs and pred_scales; defaults to test.scale_mode in config",
    )
    parser.add_argument(
        "--eval-protocol",
        type=str,
        default=None,
        choices=EVAL_PROTOCOLS,
        help=(
            "evaluation rules; defaults to test.eval_protocol or the "
            "MonoDiff9D paper protocol"
        ),
    )
    parser.add_argument(
        "--infer-t-max",
        type=int,
        default=-1,
        help="maximum diffusion timestep used during DDIM inference; -1 keeps the full schedule",
    )
    parser.add_argument(
        "--inference-init",
        type=str,
        default=None,
        choices=["random", "point_center", "bbox_point_center"],
        help="DDIM initial pose source; bbox_point_center uses depth points as a coarse pose",
    )
    parser.add_argument(
        "--coarse-timestep",
        type=int,
        default=None,
        help="starting timestep for coarse-pose refinement when --infer-t-max is not set",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="existing or target directory containing results_*.pkl",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="test split file under cfg.train_dataset.data_dir/Real; defaults to config",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="optional NOCS data root override for cfg.train_dataset.data_dir",
    )
    parser.add_argument(
        "--segmentation-results-dir",
        type=str,
        default=None,
        help=(
            "optional detection result directory containing results_*.pkl; "
            "defaults to test.segmentation_results_dir in the config"
        ),
    )
    parser.add_argument(
        "--test-rgb-enhance-mode",
        type=str,
        default="none",
        choices=["none", "brightness_contrast", "gamma", "clahe", "clahe_gamma"],
        help="optional test-time RGB crop enhancement; disabled by default to match training",
    )
    parser.add_argument(
        "--test-rgb-brightness",
        type=float,
        default=0.0,
        help="additive brightness in pixel value for test RGB enhancement",
    )
    parser.add_argument(
        "--test-rgb-contrast",
        type=float,
        default=1.0,
        help="multiplicative contrast for test RGB enhancement",
    )
    parser.add_argument(
        "--test-rgb-gamma",
        type=float,
        default=1.0,
        help="gamma for test RGB enhancement; values below 1.0 brighten dark details",
    )
    parser.add_argument(
        "--test-rgb-clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit for test RGB enhancement",
    )
    parser.add_argument(
        "--test-rgb-clahe-tile-grid-size",
        type=int,
        default=8,
        help="CLAHE tile grid size for test RGB enhancement",
    )
    parser.add_argument(
        "--test-rgb-enhance-classes",
        nargs="*",
        default=[],
        choices=["bottle", "bowl", "camera", "can", "laptop", "mug"],
        help="optional class names to enhance when test-time enhancement is explicitly enabled",
    )
    parser.add_argument(
        "--draw-bbox",
        action="store_true",
        default=False,
        help="draw bbox visualizations from results_*.pkl after inference/eval",
    )
    parser.add_argument(
        "--draw-bbox-during-test",
        action="store_true",
        default=False,
        help="draw bbox images while writing results_*.pkl; off by default to save disk space",
    )
    parser.add_argument(
        "--draw-result-path",
        type=str,
        default="",
        help="single results_*.pkl path to draw; default draws all result files",
    )
    parser.add_argument(
        "--draw-out-dir",
        type=str,
        default="",
        help="directory to save drawn bbox images; default is <save_path>/bbox_img",
    )
    parser.add_argument(
        "--draw-mode",
        type=str,
        default="3d",
        choices=["2d", "3d", "both"],
        help="bbox visualization mode",
    )
    parser.add_argument(
        "--draw-each-instance",
        action="store_true",
        default=False,
        help="draw each matched GT/pred pair into separate images",
    )
    parser.add_argument(
        "--draw-instance-index",
        type=int,
        default=None,
        help="draw only one matched pair index from each result file",
    )
    parser.add_argument(
        "--draw-suffix",
        type=str,
        default="",
        help="suffix appended to drawn image filenames",
    )
    parser.add_argument(
        "--draw-print-stats",
        action="store_true",
        default=False,
        help="print per-result translation stats while drawing",
    )
    parser.add_argument(
        "--draw-no-match-pred-to-gt",
        action="store_true",
        default=False,
        help="disable GT/pred matching before drawing and summary",
    )
    return parser.parse_args()


def bbox_iou_yxyx(box_a, box_b):
    y1 = max(box_a[0], box_b[0])
    x1 = max(box_a[1], box_b[1])
    y2 = min(box_a[2], box_b[2])
    x2 = min(box_a[3], box_b[3])
    inter = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def rotation_error_deg(pred_rt, gt_rt):
    pred_scale = np.cbrt(max(np.linalg.det(pred_rt[:3, :3]), 1.0e-12))
    gt_scale = np.cbrt(max(np.linalg.det(gt_rt[:3, :3]), 1.0e-12))
    pred_r = pred_rt[:3, :3] / max(pred_scale, 1.0e-12)
    gt_r = gt_rt[:3, :3] / max(gt_scale, 1.0e-12)
    cos_theta = (np.trace(pred_r @ gt_r.T) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _rt_rotation(pred_rt, gt_rt):
    pred_scale = np.cbrt(max(np.linalg.det(pred_rt[:3, :3]), 1.0e-12))
    gt_scale = np.cbrt(max(np.linalg.det(gt_rt[:3, :3]), 1.0e-12))
    return (
        pred_rt[:3, :3] / max(pred_scale, 1.0e-12),
        gt_rt[:3, :3] / max(gt_scale, 1.0e-12),
    )


def symmetry_aware_rotation_error_deg(pred_rt, gt_rt, class_name, handle_visibility=1):
    pred_r, gt_r = _rt_rotation(pred_rt, gt_rt)
    if class_name in ["bottle", "bowl", "can"] or (class_name == "mug" and handle_visibility == 0):
        y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_pred = pred_r @ y
        y_gt = gt_r @ y
        cos_theta = y_pred.dot(y_gt) / max(np.linalg.norm(y_pred) * np.linalg.norm(y_gt), 1.0e-12)
        return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    if class_name == "camera":
        y_180 = np.diag([-1.0, 1.0, -1.0])
        rel = pred_r @ gt_r.T
        rel_180 = pred_r @ y_180 @ gt_r.T
        cos_theta = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
        cos_theta_180 = np.clip((np.trace(rel_180) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(min(np.arccos(cos_theta), np.arccos(cos_theta_180))))
    return rotation_error_deg(pred_rt, gt_rt)


def summarize_values(name, values, logger):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        logger.warning("{}: no matched values".format(name))
        return
    logger.warning(
        "{}: mean={:.4f}, median={:.4f}, p75={:.4f}, p90={:.4f}, min={:.4f}, max={:.4f}".format(
            name,
            float(np.mean(values)),
            float(np.median(values)),
            float(np.percentile(values, 75)),
            float(np.percentile(values, 90)),
            float(np.min(values)),
            float(np.max(values)),
        )
    )


def summarize_vector_values(name, values, logger, scale=1.0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        logger.warning("{}: no matched values".format(name))
        return
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} expects shape [N, 3], got {values.shape}")
    abs_mean = np.mean(np.abs(values), axis=0) * scale
    signed_mean = np.mean(values, axis=0) * scale
    l2 = np.linalg.norm(values, axis=1) * scale
    logger.warning(
        "{}: n={} abs_xyz_mean_cm={} signed_xyz_mean_cm={} l2_mean/median_cm={:.2f}/{:.2f} l2_p90/p95/max_cm={:.2f}/{:.2f}/{:.2f}".format(
            name,
            values.shape[0],
            np.array2string(abs_mean.astype(np.float32), precision=4),
            np.array2string(signed_mean.astype(np.float32), precision=4),
            float(np.mean(l2)),
            float(np.median(l2)),
            float(np.percentile(l2, 90)),
            float(np.percentile(l2, 95)),
            float(np.max(l2)),
        )
    )


def apply_scale_mode_to_result(result, scale_mode):
    if scale_mode == "raw":
        return result

    result = dict(result)
    pred_scales = np.asarray(result["pred_scales"], dtype=np.float64)
    pred_rts = np.asarray(result["pred_RTs"], dtype=np.float64).copy()
    if scale_mode == "norm":
        pred_rt_scale = np.linalg.norm(pred_scales, axis=1, keepdims=True)
        pred_rt_scale = np.maximum(pred_rt_scale, 1.0e-6)
        pred_rts[:, :3, :3] = pred_rts[:, :3, :3] * pred_rt_scale[:, :, None]
        result["pred_RTs"] = pred_rts.astype(np.float32)
        result["pred_scales"] = (pred_scales / pred_rt_scale).astype(np.float32)
        return result

    raise ValueError(f"Unsupported scale mode: {scale_mode}")


def compute_result_stats(save_path, logger, scale_mode="raw"):
    result_paths = sorted(glob.glob(os.path.join(save_path, "results_*.pkl")))
    if not result_paths:
        raise FileNotFoundError(f"No results_*.pkl found in {save_path}")

    trans_cm = []
    rot_deg = []
    size_rel = []
    size_ratio = []
    size_l2 = []
    pred_rt_det_scale = []
    gt_rt_det_scale = []
    matched_iou = []
    coarse_self_cm = []
    coarse_pred_err_cm = []
    residual_err_cm = []
    final_minus_coarse_cm = []
    residual_z = []
    coarse_z = []
    final_z = []
    pred_coarse_to_gt_cm = []
    v3_fallback_weight = []
    v3_vote_confidence_uv = []
    v3_vote_confidence_z = []
    v3_vote_uv_rms = []
    v3_vote_log_z_rms = []
    keypoint_trans_cm = []
    keypoint_rot_deg = []
    keypoint_sym_rot_deg = []
    keypoint_residual_cm = []
    nocs_trans_cm = []
    nocs_rot_deg = []
    nocs_sym_rot_deg = []
    nocs_residual_cm = []
    nocs_valid_ratio = []
    eval_class_names = ["bottle", "bowl", "camera", "can", "laptop", "mug"]
    matched_by_class = {name: 0 for name in eval_class_names}
    per_class_final_t_cm = {name: [] for name in eval_class_names}
    per_class_coarse_to_gt_cm = {name: [] for name in eval_class_names}
    per_class_final_minus_coarse_cm = {name: [] for name in eval_class_names}
    per_class_residual_err_cm = {name: [] for name in eval_class_names}
    per_class_rot_deg = {name: [] for name in eval_class_names}
    per_class_sym_rot_deg = {name: [] for name in eval_class_names}
    per_class_keypoint_rot_deg = {name: [] for name in eval_class_names}
    per_class_keypoint_sym_rot_deg = {name: [] for name in eval_class_names}
    per_class_keypoint_t_cm = {name: [] for name in eval_class_names}
    per_class_nocs_rot_deg = {name: [] for name in eval_class_names}
    per_class_nocs_sym_rot_deg = {name: [] for name in eval_class_names}
    per_class_nocs_t_cm = {name: [] for name in eval_class_names}
    total_pred = 0
    total_gt = 0

    class_names = ["BG", "bottle", "bowl", "camera", "can", "laptop", "mug"]
    for result_path in result_paths:
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        result = apply_scale_mode_to_result(result, scale_mode)

        gt_class_ids = np.asarray(result["gt_class_ids"]).astype(np.int32)
        gt_bboxes = np.asarray(result["gt_bboxes"], dtype=np.float64)
        gt_rts = np.asarray(result["gt_RTs"], dtype=np.float64)
        gt_scales = np.asarray(result["gt_scales"], dtype=np.float64)
        pred_class_ids = np.asarray(result["pred_class_ids"]).astype(np.int32)
        pred_bboxes = np.asarray(result["pred_bboxes"], dtype=np.float64)
        pred_rts = np.asarray(result["pred_RTs"], dtype=np.float64)
        pred_scales = np.asarray(result["pred_scales"], dtype=np.float64)

        if "pred_translation_fallback_weight" in result:
            v3_fallback_weight.extend(
                np.asarray(
                    result["pred_translation_fallback_weight"],
                    dtype=np.float64,
                ).reshape(-1).tolist()
            )
        if "pred_translation_vote_confidence_mean" in result:
            confidence_mean = np.asarray(
                result["pred_translation_vote_confidence_mean"],
                dtype=np.float64,
            ).reshape(-1, 2)
            v3_vote_confidence_uv.extend(confidence_mean[:, 0].tolist())
            v3_vote_confidence_z.extend(confidence_mean[:, 1].tolist())
        if "pred_translation_vote_dispersion" in result:
            vote_dispersion = np.asarray(
                result["pred_translation_vote_dispersion"],
                dtype=np.float64,
            ).reshape(-1, 6)
            v3_vote_uv_rms.extend(
                np.linalg.norm(vote_dispersion[:, :2], axis=1).tolist()
            )
            v3_vote_log_z_rms.extend(vote_dispersion[:, 2].tolist())

        total_pred += len(pred_class_ids)
        total_gt += len(gt_class_ids)
        used_pred = set()
        for gt_idx, gt_class_id in enumerate(gt_class_ids):
            best_pred_idx = -1
            best_iou = -1.0
            for pred_idx, pred_class_id in enumerate(pred_class_ids):
                if pred_idx in used_pred or pred_class_id != gt_class_id:
                    continue
                iou = bbox_iou_yxyx(gt_bboxes[gt_idx], pred_bboxes[pred_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            if best_pred_idx < 0:
                continue

            used_pred.add(best_pred_idx)
            matched_iou.append(best_iou)
            class_name = class_names[gt_class_id] if 0 <= gt_class_id < len(class_names) else str(gt_class_id)
            if class_name in matched_by_class:
                matched_by_class[class_name] += 1

            pred_t = pred_rts[best_pred_idx, :3, 3]
            gt_t = gt_rts[gt_idx, :3, 3]
            pred_size = pred_scales[best_pred_idx]
            gt_size = gt_scales[gt_idx]
            trans_cm.append(np.linalg.norm(pred_t - gt_t) * 100.0)
            if class_name in per_class_final_t_cm:
                per_class_final_t_cm[class_name].append((pred_t - gt_t) * 100.0)
            rot_error = rotation_error_deg(pred_rts[best_pred_idx], gt_rts[gt_idx])
            sym_rot_error = symmetry_aware_rotation_error_deg(
                pred_rts[best_pred_idx],
                gt_rts[gt_idx],
                class_name,
                result["gt_handle_visibility"][gt_idx],
            )
            rot_deg.append(rot_error)
            if class_name in per_class_rot_deg:
                per_class_rot_deg[class_name].append(rot_error)
                per_class_sym_rot_deg[class_name].append(sym_rot_error)
            if "pred_keypoint_RTs" in result:
                keypoint_rts = np.asarray(result["pred_keypoint_RTs"], dtype=np.float64)
                keypoint_rt = keypoint_rts[best_pred_idx]
                keypoint_t_error = (keypoint_rt[:3, 3] - gt_t) * 100.0
                keypoint_rot_error = rotation_error_deg(keypoint_rt, gt_rts[gt_idx])
                keypoint_sym_rot_error = symmetry_aware_rotation_error_deg(
                    keypoint_rt,
                    gt_rts[gt_idx],
                    class_name,
                    result["gt_handle_visibility"][gt_idx],
                )
                keypoint_trans_cm.append(np.linalg.norm(keypoint_t_error))
                keypoint_rot_deg.append(keypoint_rot_error)
                keypoint_sym_rot_deg.append(keypoint_sym_rot_error)
                if "pred_keypoint_residual" in result:
                    keypoint_residual_cm.append(
                        float(np.asarray(result["pred_keypoint_residual"], dtype=np.float64)[best_pred_idx] * 100.0)
                    )
                if class_name in per_class_keypoint_rot_deg:
                    per_class_keypoint_rot_deg[class_name].append(keypoint_rot_error)
                    per_class_keypoint_sym_rot_deg[class_name].append(keypoint_sym_rot_error)
                    per_class_keypoint_t_cm[class_name].append(keypoint_t_error)
            if "pred_nocs_RTs" in result:
                nocs_rts = np.asarray(result["pred_nocs_RTs"], dtype=np.float64)
                nocs_rt = nocs_rts[best_pred_idx]
                nocs_t_error = (nocs_rt[:3, 3] - gt_t) * 100.0
                nocs_rot_error = rotation_error_deg(nocs_rt, gt_rts[gt_idx])
                nocs_sym_rot_error = symmetry_aware_rotation_error_deg(
                    nocs_rt,
                    gt_rts[gt_idx],
                    class_name,
                    result["gt_handle_visibility"][gt_idx],
                )
                nocs_trans_cm.append(np.linalg.norm(nocs_t_error))
                nocs_rot_deg.append(nocs_rot_error)
                nocs_sym_rot_deg.append(nocs_sym_rot_error)
                if "pred_nocs_residual" in result:
                    nocs_residual_cm.append(
                        float(np.asarray(result["pred_nocs_residual"], dtype=np.float64)[best_pred_idx] * 100.0)
                    )
                if "pred_nocs_valid_ratio" in result:
                    nocs_valid_ratio.append(
                        float(np.asarray(result["pred_nocs_valid_ratio"], dtype=np.float64)[best_pred_idx])
                    )
                if class_name in per_class_nocs_rot_deg:
                    per_class_nocs_rot_deg[class_name].append(nocs_rot_error)
                    per_class_nocs_sym_rot_deg[class_name].append(nocs_sym_rot_error)
                    per_class_nocs_t_cm[class_name].append(nocs_t_error)
            size_l2.append(np.linalg.norm(pred_size - gt_size))
            size_rel.append(np.linalg.norm(pred_size - gt_size) / max(np.linalg.norm(gt_size), 1.0e-12))
            size_ratio.extend((pred_size / np.maximum(gt_size, 1.0e-12)).tolist())
            pred_rt_det_scale.append(np.cbrt(abs(np.linalg.det(pred_rts[best_pred_idx, :3, :3]))))
            gt_rt_det_scale.append(np.cbrt(abs(np.linalg.det(gt_rts[gt_idx, :3, :3]))))
            if "pred_coarse_translation" in result:
                pred_coarse = np.asarray(result["pred_coarse_translation"], dtype=np.float64)[best_pred_idx]
                if "pred_translation_xyz" in result:
                    pred_t_xyz = np.asarray(result["pred_translation_xyz"], dtype=np.float64)[best_pred_idx]
                else:
                    pred_t_xyz = pred_t
                pred_coarse_to_gt_cm.append((pred_coarse - gt_t) * 100.0)
                final_minus_coarse_cm.append((pred_t_xyz - pred_coarse) * 100.0)
                if class_name in per_class_coarse_to_gt_cm:
                    per_class_coarse_to_gt_cm[class_name].append((pred_coarse - gt_t) * 100.0)
                    per_class_final_minus_coarse_cm[class_name].append((pred_t_xyz - pred_coarse) * 100.0)

            legacy_decode_keys = [
                "gt_bbox_coarse_translation",
                "gt_translation_residual",
                "pred_translation_residual",
                "pred_translation_xyz",
                "pred_coarse_translation",
            ]
            if (
                    result.get("translation_target_mode") == "bbox_scale_residual"
                    and all(key in result for key in legacy_decode_keys)
            ):
                gt_coarse = np.asarray(result["gt_bbox_coarse_translation"], dtype=np.float64)[gt_idx]
                pred_coarse = np.asarray(result["pred_coarse_translation"], dtype=np.float64)[best_pred_idx]
                gt_residual = np.asarray(result["gt_translation_residual"], dtype=np.float64)[gt_idx]
                pred_residual = np.asarray(result["pred_translation_residual"], dtype=np.float64)[best_pred_idx]
                pred_t_xyz = np.asarray(result["pred_translation_xyz"], dtype=np.float64)[best_pred_idx]

                coarse_self_cm.append((gt_coarse - gt_t) * 100.0)
                coarse_pred_err_cm.append((pred_coarse - gt_coarse) * 100.0)
                residual_err_cm.append((pred_residual - gt_residual) * 100.0)
                if class_name in per_class_residual_err_cm:
                    per_class_residual_err_cm[class_name].append((pred_residual - gt_residual) * 100.0)
                residual_z.append(float((pred_residual[2] - gt_residual[2]) * 100.0))
                coarse_z.append(float((pred_coarse[2] - gt_coarse[2]) * 100.0))
                final_z.append(float((pred_t_xyz[2] - gt_t[2]) * 100.0))

    logger.warning("####### Numeric Diagnostics ###################")
    logger.warning("scale mode: {}".format(scale_mode))
    logger.warning("result files: {}".format(len(result_paths)))
    logger.warning("total gt: {}, total pred: {}, matched by class+2D IoU: {}".format(total_gt, total_pred, len(trans_cm)))
    logger.warning("matched per class: {}".format(matched_by_class))
    summarize_values("2D bbox IoU of matched pairs", matched_iou, logger)
    summarize_values("translation error (cm)", trans_cm, logger)
    summarize_values("rotation error (degree)", rot_deg, logger)
    logger.warning("####### Per-Class Rotation Diagnostics ###################")
    for class_name in eval_class_names:
        if not per_class_rot_deg[class_name]:
            continue
        logger.warning("category {}".format(class_name))
        summarize_values("  raw rotation error (degree)", per_class_rot_deg[class_name], logger)
        summarize_values("  symmetry-aware rotation error (degree)", per_class_sym_rot_deg[class_name], logger)
    if keypoint_rot_deg:
        logger.warning("####### Canonical Keypoint Diagnostics ###################")
        summarize_values("keypoint-derived translation error (cm)", keypoint_trans_cm, logger)
        summarize_values("keypoint-derived raw rotation error (degree)", keypoint_rot_deg, logger)
        summarize_values("keypoint-derived symmetry-aware rotation error (degree)", keypoint_sym_rot_deg, logger)
        if keypoint_residual_cm:
            summarize_values("keypoint fit residual (cm)", keypoint_residual_cm, logger)
        logger.warning("####### Per-Class Canonical Keypoint Diagnostics ###################")
        for class_name in eval_class_names:
            if not per_class_keypoint_rot_deg[class_name]:
                continue
            logger.warning("category {}".format(class_name))
            summarize_vector_values("  keypoint-derived translation error", per_class_keypoint_t_cm[class_name], logger)
            summarize_values("  keypoint-derived raw rotation error (degree)", per_class_keypoint_rot_deg[class_name], logger)
            summarize_values(
                "  keypoint-derived symmetry-aware rotation error (degree)",
                per_class_keypoint_sym_rot_deg[class_name],
                logger,
            )
    if nocs_rot_deg:
        logger.warning("####### NOCS Correspondence Diagnostics ###################")
        summarize_values("nocs-derived translation error (cm)", nocs_trans_cm, logger)
        summarize_values("nocs-derived raw rotation error (degree)", nocs_rot_deg, logger)
        summarize_values("nocs-derived symmetry-aware rotation error (degree)", nocs_sym_rot_deg, logger)
        if nocs_residual_cm:
            summarize_values("nocs fit residual (cm)", nocs_residual_cm, logger)
        if nocs_valid_ratio:
            summarize_values("nocs valid ratio", nocs_valid_ratio, logger)
        logger.warning("####### Per-Class NOCS Correspondence Diagnostics ###################")
        for class_name in eval_class_names:
            if not per_class_nocs_rot_deg[class_name]:
                continue
            logger.warning("category {}".format(class_name))
            summarize_vector_values("  nocs-derived translation error", per_class_nocs_t_cm[class_name], logger)
            summarize_values("  nocs-derived raw rotation error (degree)", per_class_nocs_rot_deg[class_name], logger)
            summarize_values(
                "  nocs-derived symmetry-aware rotation error (degree)",
                per_class_nocs_sym_rot_deg[class_name],
                logger,
            )
    summarize_values("size L2 error", size_l2, logger)
    summarize_values("size relative L2 error", size_rel, logger)
    summarize_values("pred_size / gt_size element ratio", size_ratio, logger)
    summarize_values("det-scale from pred_RT", pred_rt_det_scale, logger)
    summarize_values("det-scale from gt_RT", gt_rt_det_scale, logger)
    if v3_fallback_weight:
        logger.warning("####### Ray-Depth V3 Diagnostics ###################")
        summarize_values("V3 fallback weight", v3_fallback_weight, logger)
        summarize_values(
            "V3 vote confidence uv", v3_vote_confidence_uv, logger
        )
        summarize_values(
            "V3 vote confidence z", v3_vote_confidence_z, logger
        )
        summarize_values("V3 vote uv RMS", v3_vote_uv_rms, logger)
        summarize_values("V3 vote log-z RMS", v3_vote_log_z_rms, logger)
    if pred_coarse_to_gt_cm:
        logger.warning("####### Coarse Translation Diagnostics ###################")
        summarize_vector_values("pred coarse vs gt translation", pred_coarse_to_gt_cm, logger)
        summarize_vector_values("pred final-minus-coarse residual", final_minus_coarse_cm, logger)
        logger.warning("####### Per-Class Coarse Translation Diagnostics ###################")
        for class_name in eval_class_names:
            if not per_class_coarse_to_gt_cm[class_name]:
                continue
            logger.warning("category {}".format(class_name))
            summarize_vector_values("  final translation error", per_class_final_t_cm[class_name], logger)
            summarize_vector_values("  pred coarse vs gt translation", per_class_coarse_to_gt_cm[class_name], logger)
            summarize_vector_values(
                "  pred final-minus-coarse residual",
                per_class_final_minus_coarse_cm[class_name],
                logger,
            )
    if coarse_self_cm:
        logger.warning("####### Translation Decode Diagnostics ###################")
        summarize_vector_values("gt coarse decode error", coarse_self_cm, logger)
        summarize_vector_values("pred coarse translation error", coarse_pred_err_cm, logger)
        summarize_vector_values("translation residual error", residual_err_cm, logger)
        summarize_vector_values("pred final-minus-coarse residual", final_minus_coarse_cm, logger)
        summarize_values("z error from gt coarse decode (cm)", [v[2] for v in coarse_self_cm], logger)
        summarize_values("z error from pred coarse translation (cm)", coarse_z, logger)
        summarize_values("z error from translation residual (cm)", residual_z, logger)
        summarize_values("z error from final translation (cm)", final_z, logger)
        logger.warning("####### Per-Class Translation Residual Diagnostics ###################")
        for class_name in eval_class_names:
            if not per_class_residual_err_cm[class_name]:
                continue
            logger.warning("category {}".format(class_name))
            summarize_vector_values("  translation residual error", per_class_residual_err_cm[class_name], logger)


class BBoxResultVisualizer:
    def __init__(
            self,
            result_dir,
            logger,
            out_dir=None,
            draw_2d=True,
            draw_3d=True,
            draw_gt=True,
            draw_pred=True,
            match_pred_to_gt=True,
            suffix=None,
            outlier_thresholds_cm=(50.0, 100.0, 1000.0),
            worst_case_top_k=20):
        self.result_dir = Path(result_dir)
        self.out_dir = Path(out_dir) if out_dir else self.result_dir / "bbox_img"
        self.logger = logger
        self.draw_2d = draw_2d
        self.draw_3d = draw_3d
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.match_pred_to_gt = match_pred_to_gt
        self.suffix = suffix or None
        self.outlier_thresholds_cm = tuple(float(v) for v in outlier_thresholds_cm)
        self.worst_case_top_k = int(worst_case_top_k)

    def draw_all(self, result_path=None, draw_each_instance=False, single_instance_index=None, print_stats=False):
        result_paths = self._get_result_paths(result_path)
        if not result_paths:
            raise FileNotFoundError(f"no results_*.pkl files found in: {self.result_dir}")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        summary = self._new_summary()

        for path in result_paths:
            img_id = self._parse_img_id(path)
            if img_id is None:
                continue

            with path.open("rb") as file:
                result = pickle.load(file)

            if print_stats:
                self._print_result_stats(path, result, matched=self.match_pred_to_gt)

            self._accumulate_summary(summary, path, result, matched=self.match_pred_to_gt)

            if draw_each_instance or single_instance_index is not None:
                saved_paths.extend(self._draw_instances(result, img_id, single_instance_index=single_instance_index))
            else:
                out_path = self._draw_result(result, img_id, self.suffix)
                if out_path is not None:
                    saved_paths.append(out_path)

        self._print_summary(summary)
        self._print_worst_cases(summary, top_k=self.worst_case_top_k)
        return saved_paths

    def _get_result_paths(self, result_path):
        if result_path:
            path = Path(result_path)
            if not path.exists():
                raise FileNotFoundError(f"result_path does not exist: {path}")
            return [path]

        if not self.result_dir.exists():
            raise FileNotFoundError(f"result_dir does not exist: {self.result_dir}")
        return sorted(self.result_dir.glob("results_*.pkl"))

    def _draw_instances(self, result, img_id, single_instance_index=None):
        if self.match_pred_to_gt:
            pairs = self._match_pairs(result)
        else:
            num_instances = min(len(result["gt_class_ids"]), len(result["pred_class_ids"]))
            pairs = [(idx, idx) for idx in range(num_instances)]

        indices = [single_instance_index] if single_instance_index is not None else range(len(pairs))
        saved_paths = []
        for idx in indices:
            if idx < 0 or idx >= len(pairs):
                continue
            gt_idx, pred_idx = pairs[idx]
            result_i = self._slice_result(result, gt_idx, pred_idx)
            suffix = f"pair_{idx}" if self.suffix is None else f"{self.suffix}_pair_{idx}"
            out_path = self._draw_result(result_i, img_id, suffix)
            if out_path is not None:
                saved_paths.append(out_path)
        return saved_paths

    def _draw_result(self, result, img_id, suffix):
        return TestingSolver._draw_box_to_image(
            None,
            result,
            str(self.out_dir),
            img_id,
            draw_2d=self.draw_2d,
            draw_3d=self.draw_3d,
            draw_gt=self.draw_gt,
            draw_pred=self.draw_pred,
            match_pred_to_gt=self.match_pred_to_gt,
            suffix=suffix,
        )

    @staticmethod
    def _slice_result(result, gt_idx, pred_idx):
        result_i = result.copy()
        for key in ["gt_class_ids", "gt_bboxes", "gt_RTs", "gt_scales", "gt_handle_visibility"]:
            if key in result:
                result_i[key] = result[key][gt_idx:gt_idx + 1]
        for key in ["pred_class_ids", "pred_bboxes", "pred_scores", "pred_RTs", "pred_scales"]:
            if key in result:
                result_i[key] = result[key][pred_idx:pred_idx + 1]
        return result_i

    def _print_result_stats(self, result_path, result, matched=False):
        gt_t = result["gt_RTs"][:, :3, 3]
        pred_t = result["pred_RTs"][:, :3, 3]
        pairs = self._match_pairs(result) if matched else [
            (idx, idx) for idx in range(min(gt_t.shape[0], pred_t.shape[0]))
        ]
        if not pairs:
            self.logger.warning("%s: no instances", result_path.name)
            return

        mode = "matched" if matched else "raw"
        self.logger.warning("%s (%s)", result_path.name, mode)
        for idx, (gt_idx, pred_idx) in enumerate(pairs):
            t_diff = pred_t[pred_idx] - gt_t[gt_idx]
            t_err_cm = np.linalg.norm(t_diff) * 100.0
            gt_scale = result["gt_scales"][gt_idx]
            pred_scale = result["pred_scales"][pred_idx]
            scale_l1 = np.abs(pred_scale - gt_scale).mean()
            self.logger.warning(
                "pair=%d gt_idx=%d pred_idx=%d gt_cls=%s pred_cls=%s t_err_cm=%.2f gt_t=%s pred_t=%s gt_scale=%s pred_scale=%s scale_l1=%.4f",
                idx,
                gt_idx,
                pred_idx,
                result["gt_class_ids"][gt_idx],
                result["pred_class_ids"][pred_idx],
                t_err_cm,
                gt_t[gt_idx],
                pred_t[pred_idx],
                gt_scale,
                pred_scale,
                scale_l1,
            )

    @staticmethod
    def _new_summary():
        return {"overall": [], "by_class": {}}

    def _accumulate_summary(self, summary, result_path, result, matched=False):
        gt_t = result["gt_RTs"][:, :3, 3]
        pred_t = result["pred_RTs"][:, :3, 3]
        pairs = self._match_pairs(result) if matched else [
            (idx, idx) for idx in range(min(gt_t.shape[0], pred_t.shape[0]))
        ]

        for gt_idx, pred_idx in pairs:
            diff_cm = (pred_t[pred_idx] - gt_t[gt_idx]) * 100.0
            gt_scale = result["gt_scales"][gt_idx]
            pred_scale = result["pred_scales"][pred_idx]
            item = {
                "result_name": result_path.name,
                "gt_idx": int(gt_idx),
                "pred_idx": int(pred_idx),
                "class_id": int(result["gt_class_ids"][gt_idx]),
                "abs_diff_cm": np.abs(diff_cm),
                "signed_diff_cm": diff_cm,
                "l2_cm": float(np.linalg.norm(diff_cm)),
                "scale_l1": float(np.abs(pred_scale - gt_scale).mean()),
            }
            summary["overall"].append(item)
            summary["by_class"].setdefault(item["class_id"], []).append(item)

    def _print_summary(self, summary):
        synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.logger.warning("==== Matched Translation Summary ====")
        self._print_summary_group("overall", summary["overall"])
        for class_id in sorted(summary["by_class"]):
            class_name = synset_names[class_id] if class_id < len(synset_names) else str(class_id)
            self._print_summary_group(class_name, summary["by_class"][class_id])

    def _print_summary_group(self, name, items):
        if not items:
            self.logger.warning("%s: no matched pairs", name)
            return

        abs_diff = np.stack([item["abs_diff_cm"] for item in items], axis=0)
        signed_diff = np.stack([item["signed_diff_cm"] for item in items], axis=0)
        l2_cm = np.asarray([item["l2_cm"] for item in items], dtype=np.float32)
        scale_l1 = np.asarray([item["scale_l1"] for item in items], dtype=np.float32)
        self.logger.warning(
            "%s: n=%d abs_xyz_mean_cm=%s signed_xyz_mean_cm=%s l2_mean/median_cm=%.2f/%.2f "
            "l2_p90/p95/max_cm=%.2f/%.2f/%.2f outliers>%.0f/%.0f/%.0fcm=%d/%d/%d scale_l1_mean=%.4f",
            name,
            len(items),
            abs_diff.mean(axis=0),
            signed_diff.mean(axis=0),
            float(l2_cm.mean()),
            float(np.median(l2_cm)),
            float(np.percentile(l2_cm, 90)),
            float(np.percentile(l2_cm, 95)),
            float(l2_cm.max()),
            self.outlier_thresholds_cm[0],
            self.outlier_thresholds_cm[1],
            self.outlier_thresholds_cm[2],
            int(np.sum(l2_cm > self.outlier_thresholds_cm[0])),
            int(np.sum(l2_cm > self.outlier_thresholds_cm[1])),
            int(np.sum(l2_cm > self.outlier_thresholds_cm[2])),
            float(scale_l1.mean()),
        )

    def _print_worst_cases(self, summary, top_k=20):
        synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        items = sorted(summary["overall"], key=lambda item: item["l2_cm"], reverse=True)
        if not items:
            self.logger.warning("==== Worst Translation Cases ====\nno matched pairs")
            return

        self.logger.warning("==== Worst Translation Cases ====")
        for item in items[:top_k]:
            class_id = item["class_id"]
            class_name = synset_names[class_id] if class_id < len(synset_names) else str(class_id)
            self.logger.warning(
                "%s class=%s gt_idx=%d pred_idx=%d l2_cm=%.2f signed_xyz_cm=%s abs_xyz_cm=%s scale_l1=%.4f",
                item["result_name"],
                class_name,
                item["gt_idx"],
                item["pred_idx"],
                item["l2_cm"],
                item["signed_diff_cm"],
                item["abs_diff_cm"],
                item["scale_l1"],
            )

    @staticmethod
    def _match_pairs(result):
        from utils.evaluation_utils import compute_3d_matches_for_each_gt

        synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        gt_class_ids = result["gt_class_ids"].astype(np.int32)
        pred_class_ids = result["pred_class_ids"].astype(np.int32)
        if len(gt_class_ids) == 0 or len(pred_class_ids) == 0:
            return []

        gt_match, pred_order = compute_3d_matches_for_each_gt(
            gt_class_ids,
            np.asarray(result["gt_RTs"]),
            np.asarray(result["gt_scales"]),
            np.asarray(result.get("gt_handle_visibility", np.ones_like(gt_class_ids))),
            synset_names,
            np.asarray(result["pred_bboxes"]),
            pred_class_ids,
            np.asarray(result.get("pred_scores", np.ones((len(pred_class_ids),), dtype=np.float32))),
            np.asarray(result["pred_RTs"]),
            np.asarray(result["pred_scales"]),
        )

        pred_order = np.asarray(pred_order, dtype=np.int64)
        pairs = []
        for gt_idx, sorted_pred_idx in enumerate(gt_match):
            if sorted_pred_idx < 0:
                continue
            pred_idx = int(pred_order[int(sorted_pred_idx)])
            pairs.append((gt_idx, pred_idx))
        return pairs

    @staticmethod
    def _parse_img_id(result_path):
        stem = result_path.stem
        prefix = "results_"
        if not stem.startswith(prefix):
            return None
        try:
            return int(stem[len(prefix):])
        except ValueError:
            return None


def infer_epoch_from_checkpoint(checkpoint_path: str) -> int:
    match = re.search(r"(?:epoch|best_epoch|last_epoch)_(\d+)", os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    return 0


def init():
    args = get_parser()
    if args.only_eval and args.inference_only:
        raise ValueError("--only-eval and --inference-only are mutually exclusive")
    if args.capture_direct_rs and not args.inference_only:
        raise ValueError("--capture-direct-rs requires --inference-only")
    exp_name = os.path.splitext(os.path.basename(args.config))[0]

    cfg = load_config(args.config)
    cfg.config_path = args.config
    cfg.gpus = args.gpus
    cfg.checkpoint = args.checkpoint or OmegaConf.select(
        cfg, "checkpoint", default=""
    )
    cfg.exp_name = exp_name
    cfg.note = args.note
    cfg.only_eval = args.only_eval
    cfg.inference_only = args.inference_only
    cfg.test.capture_direct_rs = args.capture_direct_rs
    if args.rotation_output_mode is not None:
        cfg.test.rotation_output_mode = args.rotation_output_mode
    if args.size_output_mode is not None:
        cfg.test.size_output_mode = args.size_output_mode
    if args.direct_size_blend is not None:
        if not 0.0 <= args.direct_size_blend <= 1.0:
            raise ValueError("--direct-size-blend must be in [0, 1]")
        cfg.diffusion.direct_size_blend = args.direct_size_blend
    cfg.sanity_gt = args.sanity_gt
    cfg.stats_only = args.stats_only
    cfg.scale_mode = args.scale_mode or OmegaConf.select(cfg, "test.scale_mode", default="raw")
    cfg.eval_protocol = normalize_eval_protocol(
        args.eval_protocol
        or OmegaConf.select(
            cfg,
            "test.eval_protocol",
            default=DEFAULT_EVAL_PROTOCOL,
        )
    )
    cfg.test_num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(OmegaConf.select(cfg, "test.num_workers", default=10))
    )
    if cfg.test_num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_test_images < 0:
        raise ValueError("--max-test-images must be non-negative")
    if args.test_image_offset < 0:
        raise ValueError("--test-image-offset must be non-negative")
    cfg.max_test_images = args.max_test_images
    cfg.test_image_offset = args.test_image_offset
    cfg.result_index_offset = args.test_image_offset
    cfg.denoise_sanity = args.denoise_sanity
    cfg.draw_bbox = args.draw_bbox
    cfg.draw_bbox_during_test = args.draw_bbox_during_test
    cfg.draw_result_path = args.draw_result_path
    cfg.draw_out_dir = args.draw_out_dir
    cfg.draw_mode = args.draw_mode
    cfg.draw_each_instance = args.draw_each_instance
    cfg.draw_instance_index = args.draw_instance_index
    cfg.draw_suffix = args.draw_suffix
    cfg.draw_print_stats = args.draw_print_stats
    cfg.draw_match_pred_to_gt = not args.draw_no_match_pred_to_gt
    cfg.infer_t_max = args.infer_t_max
    cfg.test_rgb_enhance_mode = args.test_rgb_enhance_mode
    cfg.test_rgb_brightness = args.test_rgb_brightness
    cfg.test_rgb_contrast = args.test_rgb_contrast
    cfg.test_rgb_gamma = args.test_rgb_gamma
    cfg.test_rgb_clahe_clip_limit = args.test_rgb_clahe_clip_limit
    cfg.test_rgb_clahe_tile_grid_size = args.test_rgb_clahe_tile_grid_size
    cfg.test_rgb_enhance_classes = list(args.test_rgb_enhance_classes)
    if args.inference_init is not None:
        cfg.diffusion.inference_init = args.inference_init
    if args.coarse_timestep is not None:
        cfg.diffusion.coarse_timestep = args.coarse_timestep

    if args.gpus != "-1":
        cfg.device = f"cuda:{args.gpus.split(',')[0]}"
        set_cuda_visible_devices(cfg.gpus)
        torch.cuda.empty_cache()
    else:
        cfg.device = "cpu"

    checkpoint_epoch = (
        infer_epoch_from_checkpoint(str(cfg.checkpoint)) if cfg.checkpoint else 0
    )
    cfg.test_epoch = checkpoint_epoch

    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_base_dir = cfg.test.test_log_dir or "testing_logs"
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_base_dir, exp_name, now_str)

    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = os.path.join(log_dir, f"eval_epoch{checkpoint_epoch}")
    cfg.save_path = save_path
    if args.data_dir:
        cfg.train_dataset.data_dir = args.data_dir
    cfg.test.split_file = args.split_file or OmegaConf.select(
        cfg, "test.split_file", default="test_list_all.txt"
    )
    cfg.segmentation_results_dir = args.segmentation_results_dir or OmegaConf.select(
        cfg, "test.segmentation_results_dir", default=None
    )

    logger = get_logger(
        level_print=logging.INFO,
        level_save=logging.INFO,
        path_file=os.path.join(log_dir, f"test_epoch{checkpoint_epoch}_logger.log"),
    )
    logger.warning("Checkpoint file path: {}".format(cfg.checkpoint))
    logger.warning("Testing Notes: {}.".format(args.note))
    return logger, cfg


def run_test(logger, cfg):
    logger.warning("\n>>>>>>>>>> Testing begins!!! <<<<<<<<<<\n")
    logger.warning(cfg)
    logger.warning("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.solver.rd_seed)
    torch.manual_seed(cfg.solver.rd_seed)

    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)

    if cfg.stats_only:
        compute_result_stats(save_path, logger, cfg.scale_mode)
    elif not cfg.only_eval:
        model = None
        if cfg.sanity_gt:
            logger.warning("=> running GT sanity check; model inference is skipped.")
        elif not cfg.checkpoint:
            raise ValueError(
                "A checkpoint is required in the config or via --checkpoint "
                "unless --only-eval is set."
            )
        else:
            logger.warning("=> creating model ...")
            if cfg.solver.model_arch != "da2_joint_geometry_pose":
                raise ValueError(
                    "This result package only supports model_arch="
                    "da2_joint_geometry_pose"
                )
            from model.da2_joint_geometry_pose import DA2JointGeometryPose
            model = DA2JointGeometryPose(cfg)

            if len(cfg.gpus) > 1:
                model.setup_data_parallel(gpu_ids=range(len(cfg.gpus.split(","))))
            model = model.to(cfg.device)

            logger.warning("=> loading checkpoint from path: {} ...".format(cfg.checkpoint))
            load_checkpoint(model, cfg.checkpoint, device=cfg.device)

            parameter_count = count_parameters(model)
            logger.warning(">>>>> Parameters for testing: {}".format(parameter_count))

        dataset = Real275TestDataset(
            base_dir=cfg.train_dataset.data_dir,
            split_file=cfg.test.split_file,
            cam_k=OmegaConf.select(cfg, "test.cam_k", default=None),
            img_size=cfg.test.img_size,
            sample_num=cfg.test.sample_num,
            rgb_enhance_mode=cfg.test_rgb_enhance_mode,
            rgb_brightness=cfg.test_rgb_brightness,
            rgb_contrast=cfg.test_rgb_contrast,
            rgb_gamma=cfg.test_rgb_gamma,
            rgb_clahe_clip_limit=cfg.test_rgb_clahe_clip_limit,
            rgb_clahe_tile_grid_size=cfg.test_rgb_clahe_tile_grid_size,
            rgb_enhance_classes=cfg.test_rgb_enhance_classes,
            depth_mode=str(OmegaConf.select(cfg, "test.depth_mode", default="legacy_estimated")),
            metric_depth_source=str(
                OmegaConf.select(cfg, "test.metric_depth.source", default="same")
            ),
            metric_depth_required=bool(
                OmegaConf.select(cfg, "test.metric_depth.required", default=True)
            ),
            metric_depth_png_scale=float(
                OmegaConf.select(cfg, "test.metric_depth.png_scale", default=1000.0)
            ),
            projection_mode=str(
                OmegaConf.select(cfg, "test.projection_mode", default="full_image")
            ),
            depth_feature_encoding=str(
                OmegaConf.select(cfg, "test.depth_feature.encoding", default="raw")
            ),
            depth_feature_min_m=float(
                OmegaConf.select(cfg, "test.depth_feature.min_m", default=0.05)
            ),
            depth_feature_max_m=float(
                OmegaConf.select(cfg, "test.depth_feature.max_m", default=5.0)
            ),
            segmentation_results_dir=cfg.segmentation_results_dir,
        )
        if cfg.test_image_offset > 0 or cfg.max_test_images > 0:
            start_index = min(int(cfg.test_image_offset), len(dataset))
            stop_index = len(dataset)
            if cfg.max_test_images > 0:
                stop_index = min(start_index + int(cfg.max_test_images), len(dataset))
            dataset = torch.utils.data.Subset(dataset, range(start_index, stop_index))
            logger.warning(
                "=> limiting inference to test-image range [%d, %d)",
                start_index,
                stop_index,
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.test_num_workers,
            pin_memory=cfg.train_dataloader.pin_memory,
            drop_last=False,
            collate_fn=test_collate_fn,
        )

        tester = TestingSolver(model=model, save_path=save_path, dataloader=dataloader, cfg=cfg, logger=logger)
        if cfg.denoise_sanity:
            tester.denoise_sanity()
        else:
            tester.test()

    if not cfg.stats_only and not cfg.denoise_sanity and not cfg.inference_only:
        evaluate(save_path, logger, eval_protocol=cfg.eval_protocol)
        compute_result_stats(save_path, logger, cfg.scale_mode)
    if cfg.draw_bbox and not cfg.denoise_sanity:
        visualizer = BBoxResultVisualizer(
            result_dir=save_path,
            out_dir=cfg.draw_out_dir,
            logger=logger,
            draw_2d=cfg.draw_mode in ("2d", "both"),
            draw_3d=cfg.draw_mode in ("3d", "both"),
            draw_gt=True,
            draw_pred=True,
            match_pred_to_gt=cfg.draw_match_pred_to_gt,
            suffix=cfg.draw_suffix,
        )
        saved_paths = visualizer.draw_all(
            result_path=cfg.draw_result_path,
            draw_each_instance=cfg.draw_each_instance,
            single_instance_index=cfg.draw_instance_index,
            print_stats=cfg.draw_print_stats,
        )
        logger.warning("saved %d bbox images to: %s", len(saved_paths), visualizer.out_dir)

    logger.warning("\n>>>>>>>>>> Testing Completed!!! <<<<<<<<<<\n")


def run_with_notifications():
    logger, cfg = init()
    start_msg = (
        f"GeoQueryPose test started\n"
        f"exp: {cfg.exp_name}\n"
        f"checkpoint: {cfg.checkpoint or 'none'}\n"
        f"save_path: {cfg.save_path}"
    )
    notify_test_start(start_msg)

    try:
        run_test(logger, cfg)
        notify_test_end(
            f"GeoQueryPose test finished\nexp: {cfg.exp_name}\nsave_path: {cfg.save_path}",
            success=True,
        )
    except BaseException as exc:
        notify_test_end(
            f"GeoQueryPose test failed\nexp: {cfg.exp_name}\nsave_path: {cfg.save_path}\nerror: {exc}",
            success=False,
        )
        raise


if __name__ == "__main__":
    run_with_notifications()
