

""" Modified based on https://github.com/hughw19/NOCS_CVPR2019."""
import logging
import os
import sys
import numpy as np
import glob
import math
import _pickle as cPickle
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
from ctypes import *
import math


EVAL_PROTOCOL_MONODIFF9D_PAPER = "monodiff9d_paper"
EVAL_PROTOCOL_CORRECTED = "corrected"
DEFAULT_EVAL_PROTOCOL = EVAL_PROTOCOL_MONODIFF9D_PAPER
EVAL_PROTOCOLS = (
    EVAL_PROTOCOL_MONODIFF9D_PAPER,
    EVAL_PROTOCOL_CORRECTED,
)


def normalize_eval_protocol(eval_protocol=None):
    protocol = eval_protocol or DEFAULT_EVAL_PROTOCOL
    if protocol not in EVAL_PROTOCOLS:
        raise ValueError(
            f"Unsupported evaluation protocol: {protocol}. "
            f"Expected one of {EVAL_PROTOCOLS}."
        )
    return protocol


def trim_zeros(x):
    """Run trim zeros.

    Args:
        x: Input tensor.

    Returns:
        Function output.
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x

def get_3d_bbox(scale, shift=0):
    """Get 3d bbox.

    Args:
        scale: Input parameter.
        shift: Input parameter.

    Returns:
        Requested value.
    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                            [scale / 2, +scale / 2, -scale / 2],
                            [-scale / 2, +scale / 2, scale / 2],
                            [-scale / 2, +scale / 2, -scale / 2],
                            [+scale / 2, -scale / 2, scale / 2],
                            [+scale / 2, -scale / 2, -scale / 2],
                            [-scale / 2, -scale / 2, scale / 2],
                            [-scale / 2, -scale / 2, -scale / 2]]) + shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def transform_coordinates_3d(coordinates, RT):
    """Run transform coordinates 3d.

    Args:
        coordinates: Input parameter.
        RT: Input parameter.

    Returns:
        Function output.
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones(
        (1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match):

    """Compute ap from matches scores.

    Args:
        pred_match: Predicted value.
        pred_scores: Predicted value.
        gt_match: Ground-truth value.

    Returns:
        Computed result.
    """
    assert pred_match.shape[0] == pred_scores.shape[0]

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1])
                * precisions[indices])
    return ap

def _compute_3d_iou_monodiff9d_paper(
    RT_1,
    RT_2,
    scales_1,
    scales_2,
    handle_visibility,
    class_name_1,
    class_name_2,
):
    """Reproduce the 3D-IoU implementation used by MonoDiff9D's paper code."""

    def asymmetric_3d_iou(first_rt, second_rt, first_scale, second_scale):
        first_bbox = transform_coordinates_3d(get_3d_bbox(first_scale, 0), first_rt)
        second_bbox = transform_coordinates_3d(get_3d_bbox(second_scale, 0), second_rt)

        # axis=0 intentionally matches the released MonoDiff9D evaluator.
        first_max = np.amax(first_bbox, axis=0)
        first_min = np.amin(first_bbox, axis=0)
        second_max = np.amax(second_bbox, axis=0)
        second_min = np.amin(second_bbox, axis=0)

        overlap_min = np.maximum(first_min, second_min)
        overlap_max = np.minimum(first_max, second_max)
        if np.amin(overlap_max - overlap_min) < 0:
            intersection = 0
        else:
            intersection = np.prod(overlap_max - overlap_min)
        union = (
            np.prod(first_max - first_min)
            + np.prod(second_max - second_min)
            - intersection
        )
        return intersection / union

    if RT_1 is None or RT_2 is None:
        return -1

    is_symmetric = (
        class_name_1 in ["bottle", "bowl", "can"]
        and class_name_1 == class_name_2
    ) or (
        class_name_1 == "mug"
        and class_name_1 == class_name_2
        and handle_visibility == 0
    )
    if not is_symmetric:
        return asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

    max_iou = 0
    for index in range(20):
        theta = 2 * math.pi * index / 20.0
        y_rotation = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ])
        max_iou = max(
            max_iou,
            asymmetric_3d_iou(
                RT_1 @ y_rotation,
                RT_2,
                scales_1,
                scales_2,
            ),
        )
    return max_iou


def compute_3d_iou_new(
    RT_1,
    RT_2,
    scales_1,
    scales_2,
    handle_visibility,
    class_name_1,
    class_name_2,
    eval_protocol=DEFAULT_EVAL_PROTOCOL,
):
    """Compute 3d iou new.

    Args:
        RT_1: Input parameter.
        RT_2: Input parameter.
        scales_1: Input parameter.
        scales_2: Input parameter.
        handle_visibility: Input parameter.
        class_name_1: Input parameter.
        class_name_2: Input parameter.

    Returns:
        Computed result.
    """

    eval_protocol = normalize_eval_protocol(eval_protocol)
    if eval_protocol == EVAL_PROTOCOL_MONODIFF9D_PAPER:
        return _compute_3d_iou_monodiff9d_paper(
            RT_1,
            RT_2,
            scales_1,
            scales_2,
            handle_visibility,
            class_name_1,
            class_name_2,
        )

    def should_handle_symmetry(class_name, handle_visibility):
        """Run should handle symmetry.

        Args:
            class_name: Input parameter.
            handle_visibility: Input parameter.

        Returns:
            Function output.
        """
        symmetric_classes = {
            'bottle': ['y'],
            'bowl'  : ['y'],
            'can'   : ['y'],
            'mug'   : ['y'],

            'camera': [],
            'laptop': []
        }
        return symmetric_classes.get(class_name, [])

    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        """Run asymmetric 3d iou.

        Args:
            RT_1: Input parameter.
            RT_2: Input parameter.
            scales_1: Input parameter.
            scales_2: Input parameter.

        Returns:
            Function output.
        """
        try:
            noc_cube_1 = get_3d_bbox(scales_1, 0)
            bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

            noc_cube_2 = get_3d_bbox(scales_2, 0)
            bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

            bbox_1_max = np.amax(bbox_3d_1, axis=1)
            bbox_1_min = np.amin(bbox_3d_1, axis=1)
            bbox_2_max = np.amax(bbox_3d_2, axis=1)
            bbox_2_min = np.amin(bbox_3d_2, axis=1)

            overlap_min = np.maximum(bbox_1_min, bbox_2_min)
            overlap_max = np.minimum(bbox_1_max, bbox_2_max)

            overlap_dims = overlap_max - overlap_min
            if np.any(overlap_dims < 0):
                intersections = 0.0
            else:
                intersections = np.prod(np.maximum(overlap_dims, 0.0))

            union = np.prod(bbox_1_max - bbox_1_min) + \
                    np.prod(bbox_2_max - bbox_2_min) - intersections

            if union <= 0:
                return 0.0

            overlaps = intersections / union
            return max(0.0, min(1.0, overlaps))

        except Exception as e:
            print(f"3D IoU error: {e}")
            return 0.0

    def y_rotation_matrix(theta):
        """Run y rotation matrix.

        Args:
            theta: Input parameter.

        Returns:
            Function output.
        """
        return np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    def x_rotation_matrix(theta):
        """Run x rotation matrix.

        Args:
            theta: Input parameter.

        Returns:
            Function output.
        """
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    def z_rotation_matrix(theta):
        """Run z rotation matrix.

        Args:
            theta: Input parameter.

        Returns:
            Function output.
        """
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    if RT_1 is None or RT_2 is None:
        return -1

    symmetry_axes = should_handle_symmetry(class_name_1, handle_visibility)

    if symmetry_axes and class_name_1 == class_name_2:
        max_iou = 0
        n = 72

        for axis in symmetry_axes:
            if axis == 'y':
                rotation_fn = y_rotation_matrix
            elif axis == 'x':
                rotation_fn = x_rotation_matrix
            elif axis == 'z':
                rotation_fn = z_rotation_matrix
            else:
                continue

            for i in range(n):
                rotated_RT_1 = RT_1 @ rotation_fn(2 * math.pi * i / float(n))
                iou = asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2)
                max_iou = max(max_iou, iou)
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

    return max_iou

def compute_3d_iou_old(RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2):
    """Compute 3d iou old.

    Args:
        RT_1: Input parameter.
        RT_2: Input parameter.
        scales_1: Input parameter.
        scales_2: Input parameter.
        handle_visibility: Input parameter.
        class_name_1: Input parameter.
        class_name_2: Input parameter.

    Returns:
        Computed result.
    """

    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        """Run asymmetric 3d iou.

        Args:
            RT_1: Input parameter.
            RT_2: Input parameter.
            scales_1: Input parameter.
            scales_2: Input parameter.

        Returns:
            Function output.
        """
        noc_cube_1 = get_3d_bbox(scales_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=1)
        bbox_1_min = np.amin(bbox_3d_1, axis=1)
        bbox_2_max = np.amax(bbox_3d_2, axis=1)
        bbox_2_min = np.amin(bbox_3d_2, axis=1)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + \
            np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if RT_1 is None or RT_2 is None:
        return -1

    symmetry_flag = False
    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or (
            class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility == 0):

        noc_cube_1 = get_3d_bbox(scales_1, 0)
        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        def y_rotation_matrix(theta):
            """Run y rotation matrix.

            Args:
                theta: Input parameter.

            Returns:
                Function output.
            """
            return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0, 0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0, 0, 0, 1]])

        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = RT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
            max_iou = max(max_iou,
                          asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2))
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

    return max_iou

def compute_combination_RT_degree_cm_symmetry(RT_1, RT_2, scale, class_id, handle_visibility, synset_names):
    """Compute combination rt degree cm symmetry.

    Args:
        RT_1: Input parameter.
        RT_2: Input parameter.
        scale: Input parameter.
        class_id: Identifier value.
        handle_visibility: Input parameter.
        synset_names: Input parameter.

    Returns:
        Computed result.
    """

    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    if synset_names[class_id] in ['bottle', 'can', 'bowl']:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))

    elif synset_names[class_id] == 'mug' and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) / scale
    result = np.array([theta, shift])

    return result

def compute_combination_3d_matches(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                                   pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                                   iou_3d_thresholds, degree_thesholds, shift_thesholds, score_threshold=0):
    """Compute combination 3d matches.

    Args:
        gt_class_ids: Ground-truth value.
        gt_RTs: Ground-truth value.
        gt_scales: Ground-truth value.
        gt_handle_visibility: Ground-truth value.
        synset_names: Input parameter.
        pred_boxes: Predicted value.
        pred_class_ids: Predicted value.
        pred_scores: Predicted value.
        pred_RTs: Predicted value.
        pred_scales: Predicted value.
        iou_3d_thresholds: Input parameter.
        degree_thesholds: Input parameter.
        shift_thesholds: Input parameter.
        score_threshold: Input parameter.

    Returns:
        Computed result.
    """

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    RT_overlaps = np.zeros((num_pred, num_gt, 2), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):

            overlaps[i, j] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j],
                                                gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])

            RT_overlaps[i, j, :] = compute_combination_RT_degree_cm_symmetry(pred_RTs[i], gt_RTs[j], np.cbrt(
                np.linalg.det(gt_RTs[j, :3, :3])), gt_class_ids[j], gt_handle_visibility[j], synset_names)

    num_iou_3d_thres = len(iou_3d_thresholds)
    num_degree_thes = len(degree_thesholds)
    num_shift_thes = len(shift_thesholds)
    pred_matches = -1 * \
        np.ones([num_degree_thes, num_shift_thes, num_iou_3d_thres, num_pred])
    gt_matches = -1 * \
        np.ones([num_degree_thes, num_shift_thes, num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for d, degree_thres in enumerate(degree_thesholds):
            for t, shift_thres in enumerate(shift_thesholds):
                for i in range(len(pred_boxes)):

                    sorted_ixs_by_iou = np.argsort(overlaps[i])[::-1]

                    low_score_idx = np.where(
                        overlaps[i, sorted_ixs_by_iou] < score_threshold)[0]
                    if low_score_idx.size > 0:
                        sorted_ixs_by_iou = sorted_ixs_by_iou[:low_score_idx[0]]

                    for j in sorted_ixs_by_iou:
                        if gt_matches[d, t, s, j] > -1:
                            continue

                        iou = overlaps[i, j]
                        r_error = RT_overlaps[i, j, 0]
                        t_error = RT_overlaps[i, j, 1]

                        if iou < iou_thres or r_error > degree_thres or t_error > shift_thres:
                            break

                        if not pred_class_ids[i] == gt_class_ids[j]:
                            continue

                        if iou >= iou_thres or r_error <= degree_thres or t_error <= shift_thres:
                            gt_matches[d, t, s, j] = i
                            pred_matches[d, t, s, i] = j
                            break

    return gt_matches, pred_matches, indices

def compute_combination_mAP(final_results, synset_names, degree_thresholds=[5, 10, 15], shift_thresholds=[0.1, 0.2], iou_3d_thresholds=[0.1]):
    """Compute combination m ap.

    Args:
        final_results: Input parameter.
        synset_names: Input parameter.
        degree_thresholds: Input parameter.
        shift_thresholds: Input parameter.
        iou_3d_thresholds: Input parameter.

    Returns:
        Computed result.
    """
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    aps = np.zeros((num_classes + 1, num_degree_thres,
                    num_shift_thres, num_iou_thres))
    pred_matches_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]
    gt_matches_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]
    pred_scores_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]

    for progress, result in tqdm(enumerate(final_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):

            cls_gt_class_ids = gt_class_ids[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 4, 4))

            cls_pred_class_ids = pred_class_ids[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_bboxes = pred_bboxes[pred_class_ids == cls_id, :] if len(
                pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 3))

            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids == cls_id] if len(
                    gt_class_ids) else np.ones(0)

            gt_match, pred_match, pred_indiced = compute_combination_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                iou_thres_list, degree_thres_list, shift_thres_list)
            if len(pred_indiced):
                cls_pred_class_ids = cls_pred_class_ids[pred_indiced]
                cls_pred_RTs = cls_pred_RTs[pred_indiced]
                cls_pred_scores = cls_pred_scores[pred_indiced]
                cls_pred_bboxes = cls_pred_bboxes[pred_indiced]

            pred_matches_all[cls_id] = np.concatenate(
                (pred_matches_all[cls_id], pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(
                cls_pred_scores, (num_degree_thres, num_shift_thres, num_iou_thres, 1))
            pred_scores_all[cls_id] = np.concatenate(
                (pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert pred_matches_all[cls_id].shape[-1] == pred_scores_all[cls_id].shape[-1]
            gt_matches_all[cls_id] = np.concatenate(
                (gt_matches_all[cls_id], gt_match), axis=-1)

    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        for s, iou_thres in enumerate(iou_thres_list):
            for d, degree_thres in enumerate(degree_thres_list):
                for t, shift_thres in enumerate(shift_thres_list):
                    aps[cls_id, d, t, s] = compute_ap_from_matches_scores(pred_matches_all[cls_id][d, t, s, :],
                                                                          pred_scores_all[cls_id][d,
                                                                                                  t, s, :],
                                                                          gt_matches_all[cls_id][d, t, s, :])

    aps[-1, :, :, :] = np.mean(aps[1:-1, :, :, :], axis=0)

    print('IoU75, 5  degree,  5% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    print('IoU75, 10 degree,  5% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    print('IoU75, 5  degree, 10% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.10), iou_thres_list.index(0.75)]*100))
    print('IoU50, 5  degree, 20% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))
    print('IoU50, 10 degree, 10% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.10), iou_thres_list.index(0.50)]*100))
    print('IoU50, 10 degree, 20% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))

    return aps

def compute_3d_matches_for_each_gt(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                       pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales):

    """Compute 3d matches for each gt.

    Args:
        gt_class_ids: Ground-truth value.
        gt_RTs: Ground-truth value.
        gt_scales: Ground-truth value.
        gt_handle_visibility: Ground-truth value.
        synset_names: Input parameter.
        pred_boxes: Predicted value.
        pred_class_ids: Predicted value.
        pred_scores: Predicted value.
        pred_RTs: Predicted value.
        pred_scales: Predicted value.

    Returns:
        Computed result.
    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    overlaps = np.zeros((num_gt, num_pred), dtype=np.float32)

    for j in range(num_gt):
        for i in range(num_pred):

            overlaps[j, i] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j],
                                                gt_handle_visibility[j], synset_names[pred_class_ids[i].squeeze()], synset_names[gt_class_ids[j].squeeze()])

    pred_matches = -1 * np.ones([num_pred, ])
    gt_matches = -1 * np.ones([num_gt,], dtype=np.int32)

    for i in range(num_gt):
        sorted_ixs = np.argsort(overlaps[i])[::-1]

        for j in sorted_ixs:
            if pred_matches[j] > -1:
                continue

            if not pred_class_ids[j] == gt_class_ids[i]:
                continue

            gt_matches[i] = j
            pred_matches[j] = i
            break

    return gt_matches, indices

def compute_3d_matches(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                       pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                       iou_3d_thresholds, score_threshold=0,
                       eval_protocol=DEFAULT_EVAL_PROTOCOL):
    """Compute 3d matches.

    Args:
        gt_class_ids: Ground-truth value.
        gt_RTs: Ground-truth value.
        gt_scales: Ground-truth value.
        gt_handle_visibility: Ground-truth value.
        synset_names: Input parameter.
        pred_boxes: Predicted value.
        pred_class_ids: Predicted value.
        pred_scores: Predicted value.
        pred_RTs: Predicted value.
        pred_scales: Predicted value.
        iou_3d_thresholds: Input parameter.
        score_threshold: Input parameter.

    Returns:
        Computed result.
    """

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):

            overlaps[i, j] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j],
                                                gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]],
                                                eval_protocol=eval_protocol)

    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(len(pred_boxes)):

            sorted_ixs = np.argsort(overlaps[i])[::-1]

            low_score_idx = np.where(
                overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]

            for j in sorted_ixs:

                if gt_matches[s, j] > -1:
                    continue

                iou = overlaps[i, j]

                if iou < iou_thres:
                    break

                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    return gt_matches, pred_matches, overlaps, indices

def compute_RT_degree_cm_symmetry(
    RT_1,
    RT_2,
    class_id,
    handle_visibility,
    synset_names,
    eval_protocol=DEFAULT_EVAL_PROTOCOL,
):
    """Compute rt degree cm symmetry.

    Args:
        RT_1: Input parameter.
        RT_2: Input parameter.
        class_id: Identifier value.
        handle_visibility: Input parameter.
        synset_names: Input parameter.

    Returns:
        Computed result.
    """

    eval_protocol = normalize_eval_protocol(eval_protocol)
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    class_name = synset_names[class_id]
    paper_axial_symmetry = class_name in ['bottle', 'can', 'bowl'] or (
        class_name == 'mug' and handle_visibility == 0
    )
    corrected_axial_symmetry = class_name in ['bottle', 'can', 'bowl', 'mug']

    if (
        eval_protocol == EVAL_PROTOCOL_MONODIFF9D_PAPER
        and paper_axial_symmetry
    ) or (
        eval_protocol == EVAL_PROTOCOL_CORRECTED
        and corrected_axial_symmetry
    ):
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))

    elif eval_protocol == EVAL_PROTOCOL_CORRECTED and class_name == 'camera':
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)),
                    np.arccos(np.clip((np.trace(R_rot) - 1) / 2, -1.0, 1.0)))
    elif class_name in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result

def compute_RT_overlaps(gt_class_ids, gt_RTs, gt_handle_visibility,
                        pred_class_ids, pred_RTs,
                        synset_names,
                        eval_protocol=DEFAULT_EVAL_PROTOCOL):
    """Compute rt overlaps.

    Args:
        gt_class_ids: Ground-truth value.
        gt_RTs: Ground-truth value.
        gt_handle_visibility: Ground-truth value.
        pred_class_ids: Predicted value.
        pred_RTs: Predicted value.
        synset_names: Input parameter.

    Returns:
        Computed result.
    """

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_degree_cm_symmetry(pred_RTs[i],
                                                              gt_RTs[j],
                                                              gt_class_ids[j],
                                                              gt_handle_visibility[j],
                                                              synset_names,
                                                              eval_protocol=eval_protocol)

    return overlaps

def compute_match_from_degree_cm(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    """Compute match from degree cm.

    Args:
        overlaps: Input parameter.
        pred_class_ids: Predicted value.
        gt_class_ids: Ground-truth value.
        degree_thres_list: Input list.
        shift_thres_list: Input list.

    Returns:
        Computed result.
    """
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):

                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)

                for j in sorted_ixs:

                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue

                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue

                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches

def compute_independent_mAP(final_results, synset_names, degree_thresholds=[360], shift_thresholds=[100], iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=True, logger=None, plot_figure=True, log_dir=None, eval_protocol=DEFAULT_EVAL_PROTOCOL):

    """Compute independent m ap.

    Args:
        final_results: Input parameter.
        synset_names: Input parameter.
        degree_thresholds: Input parameter.
        shift_thresholds: Input parameter.
        iou_3d_thresholds: Input parameter.
        iou_pose_thres: Input parameter.
        use_matches_for_pose: Boolean flag.
        logger: Logger instance.
        plot_figure: Input parameter.
        log_dir: Directory path.

    Returns:
        Computed result.
    """
    eval_protocol = normalize_eval_protocol(eval_protocol)
    if logger is not None:
        logger.warning("Evaluation protocol: %s", eval_protocol)

    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 0))
                            for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 0))
                           for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 0))
                          for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]
    pose_gt_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]
    pose_pred_scores_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]

    progress = 0
    for progress, result in tqdm(enumerate(final_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):

            cls_gt_class_ids = gt_class_ids[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 4, 4))

            cls_pred_class_ids = pred_class_ids[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_bboxes = pred_bboxes[pred_class_ids == cls_id, :] if len(
                pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 3))

            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids == cls_id] if len(
                    gt_class_ids) else np.ones(0)

            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                           cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                           iou_thres_list,
                                                                                           eval_protocol=eval_protocol)

            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]

            iou_pred_matches_all[cls_id] = np.concatenate(
                (iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_all[cls_id] = np.concatenate(
                (iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
            iou_gt_matches_all[cls_id] = np.concatenate(
                (iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros((0, 4))

                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros(0)

            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_RTs, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_RTs,
                                              synset_names,
                                              eval_protocol=eval_protocol)

            pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(RT_overlaps,
                                                                                  cls_pred_class_ids,
                                                                                  cls_gt_class_ids,
                                                                                  degree_thres_list,
                                                                                  shift_thres_list)

            pose_pred_matches_all[cls_id] = np.concatenate(
                (pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1)

            cls_pred_scores_tile = np.tile(
                cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            pose_pred_scores_all[cls_id] = np.concatenate(
                (pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert pose_pred_scores_all[cls_id].shape[2] == pose_pred_matches_all[cls_id].shape[2], '{} vs. {}'.format(
                pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape)
            pose_gt_matches_all[cls_id] = np.concatenate(
                (pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1)

    fig_iou = plt.figure(figsize=(30,10))

    ax_iou = plt.subplot(131)
    plt.ylabel('AP')
    plt.ylim((0, 1))
    plt.tick_params(labelsize=20)
    plt.xlabel('3D IoU thresholds', fontsize=24)

    iou_dict = {}
    iou_dict['thres_list'] = iou_thres_list
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        for s, iou_thres in enumerate(iou_thres_list):
            iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
                                                                   iou_pred_scores_all[cls_id][s, :],
                                                                   iou_gt_matches_all[cls_id][s, :])
        ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)

    iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
    ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
    iou_dict['aps'] = iou_3d_aps

    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all,
                                                                        cls_pose_pred_scores_all,
                                                                        cls_pose_gt_matches_all)

            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])

    ax_trans = plt.subplot(132)
    plt.ylim((0, 1))
    plt.tick_params(labelsize=20)
    plt.xlabel('Rotation/degree', fontsize=24)
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]

        ax_trans.plot(
            degree_thres_list[:-1], pose_aps[cls_id, :-1, -1], label=class_name)

    ax_trans.plot(degree_thres_list[:-1], pose_aps[-1, :-1, -1], label='mean')

    ax_rot = plt.subplot(133)
    plt.ylim((0, 1))
    plt.tick_params(labelsize=20)
    plt.xlabel('translation/cm', fontsize=24)
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]

        ax_rot.plot(shift_thres_list[:-1],
                    pose_aps[cls_id, -1, :-1], label=class_name)

    ax_rot.plot(shift_thres_list[:-1], pose_aps[-1, -1, :-1], label='mean')

    plt.legend(loc='lower right')

    plot_save_path = os.path.join(log_dir, 'visual')
    if not os.path.isdir(plot_save_path):
        os.mkdir(plot_save_path)

    output_path = os.path.join(
        plot_save_path, 'mAP_{}-{}cm.png'.format(shift_thres_list[0], shift_thres_list[-2]))
    ax_rot.legend()

    if plot_figure:
        fig_iou.savefig(output_path)
    plt.close(fig_iou)

    if logger is not None:
        logger.warning('3D IoU at 25: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.25)] * 100))
        logger.warning('3D IoU at 50: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.5)] * 100))
        logger.warning('3D IoU at 75: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.75)] * 100))

        logger.warning('5 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(2)] * 100))
        logger.warning('5 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(5)] * 100))

        logger.warning('10 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(2)] * 100))
        logger.warning('10 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(5)] * 100))
        logger.warning('10 degree, 10cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(10)] * 100))
        logger.warning('10 degree: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), -1] * 100))
        logger.warning('10 cm: {:.1f}'.format(
            pose_aps[-1, -1, shift_thres_list.index(10)] * 100))

        logger.warning('####### Per Class result ###################')
        for idx in range(1, len(synset_names)):
            logger.warning('category {}'.format(synset_names[idx]))
            logger.warning('mAP:')
            logger.warning('3D IoU at 25: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.25)] * 100))
            logger.warning('3D IoU at 50: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.5)] * 100))
            logger.warning('3D IoU at 75: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.75)] * 100))
            logger.warning('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(5), shift_thres_list.index(2)] * 100))
            logger.warning('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(5), shift_thres_list.index(5)] * 100))
            logger.warning('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(2)] * 100))
            logger.warning('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(5)] * 100))
            logger.warning('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(10)] * 100))
            logger.warning('10 degree: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), -1] * 100))
            logger.warning('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_thres_list.index(10)] * 100))

    return iou_3d_aps, pose_aps

_ADD_S_MODEL_POINT_CACHE = {}


def _normalize_rt_rotation(RT):
    """Return RT with the uniform scale removed from the rotation block."""
    clean_RT = np.asarray(RT, dtype=np.float64).copy()
    det = np.linalg.det(clean_RT[:3, :3])
    if np.isfinite(det) and abs(det) > 1.0e-12:
        clean_RT[:3, :3] = clean_RT[:3, :3] / np.cbrt(det)
    return clean_RT


def _cuboid_model_points(scale, samples_per_axis=10):
    """Sample deterministic surface points from a cuboid with the given metric scale."""
    scale = np.asarray(scale, dtype=np.float64).reshape(3)
    cache_key = (tuple(np.round(scale, 6)), int(samples_per_axis))
    if cache_key in _ADD_S_MODEL_POINT_CACHE:
        return _ADD_S_MODEL_POINT_CACHE[cache_key]

    xs = np.linspace(-scale[0] / 2.0, scale[0] / 2.0, samples_per_axis)
    ys = np.linspace(-scale[1] / 2.0, scale[1] / 2.0, samples_per_axis)
    zs = np.linspace(-scale[2] / 2.0, scale[2] / 2.0, samples_per_axis)

    points = []
    for x in (xs[0], xs[-1]):
        yy, zz = np.meshgrid(ys, zs, indexing="ij")
        points.append(np.stack([np.full_like(yy, x), yy, zz], axis=-1).reshape(-1, 3))
    for y in (ys[0], ys[-1]):
        xx, zz = np.meshgrid(xs, zs, indexing="ij")
        points.append(np.stack([xx, np.full_like(xx, y), zz], axis=-1).reshape(-1, 3))
    for z in (zs[0], zs[-1]):
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        points.append(np.stack([xx, yy, np.full_like(xx, z)], axis=-1).reshape(-1, 3))

    model_points = np.unique(np.concatenate(points, axis=0), axis=0)
    _ADD_S_MODEL_POINT_CACHE[cache_key] = model_points
    return model_points


def _transform_points(points, RT):
    clean_RT = _normalize_rt_rotation(RT)
    return points @ clean_RT[:3, :3].T + clean_RT[:3, 3]


def compute_adds_error(pred_RT, gt_RT, gt_scale, samples_per_axis=10):
    """Compute ADD-S mean nearest-neighbor distance in meters."""
    model_points = _cuboid_model_points(gt_scale, samples_per_axis=samples_per_axis)
    pred_points = _transform_points(model_points, pred_RT)
    gt_points = _transform_points(model_points, gt_RT)

    min_distances = []
    chunk_size = 128
    for start in range(0, pred_points.shape[0], chunk_size):
        chunk = pred_points[start:start + chunk_size]
        distances = np.linalg.norm(chunk[:, None, :] - gt_points[None, :, :], axis=-1)
        min_distances.append(np.min(distances, axis=1))
    return float(np.mean(np.concatenate(min_distances, axis=0)))


def compute_adds_matches(gt_class_ids, gt_RTs, gt_scales, pred_class_ids, pred_scores, pred_RTs,
                         synset_names, threshold_ratios=(0.02, 0.05, 0.1), samples_per_axis=10):
    """Greedily match predictions to GT using ADD-S thresholds normalized by object diameter."""
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    threshold_ratios = list(threshold_ratios)
    num_thresholds = len(threshold_ratios)

    pred_matches = -1 * np.ones((num_thresholds, num_pred), dtype=np.int32)
    gt_matches = -1 * np.ones((num_thresholds, num_gt), dtype=np.int32)
    adds_errors = np.full((num_pred, num_gt), np.inf, dtype=np.float64)
    norm_errors = np.full((num_pred, num_gt), np.inf, dtype=np.float64)

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches, adds_errors, norm_errors, np.arange(num_pred)

    indices = np.argsort(pred_scores)[::-1]
    pred_class_ids = pred_class_ids[indices].copy()
    pred_scores = pred_scores[indices].copy()
    pred_RTs = pred_RTs[indices].copy()

    for i in range(num_pred):
        for j in range(num_gt):
            if pred_class_ids[i] != gt_class_ids[j]:
                continue
            diameter = float(np.linalg.norm(gt_scales[j]))
            if diameter <= 1.0e-12:
                continue
            adds_errors[i, j] = compute_adds_error(
                pred_RTs[i],
                gt_RTs[j],
                gt_scales[j],
                samples_per_axis=samples_per_axis,
            )
            norm_errors[i, j] = adds_errors[i, j] / diameter

    for t, threshold_ratio in enumerate(threshold_ratios):
        for i in range(num_pred):
            sorted_gt = np.argsort(norm_errors[i])
            for j in sorted_gt:
                if not np.isfinite(norm_errors[i, j]):
                    break
                if gt_matches[t, j] > -1:
                    continue
                if norm_errors[i, j] > threshold_ratio:
                    break
                gt_matches[t, j] = i
                pred_matches[t, i] = j
                break

    return gt_matches, pred_matches, adds_errors, norm_errors, indices


def compute_adds_mAP(final_results, synset_names, threshold_ratios=(0.02, 0.05, 0.1),
                     logger=None, samples_per_axis=10):
    """Compute ADD-S AP and matched-pair accuracy for saved evaluation results."""
    num_classes = len(synset_names)
    threshold_ratios = list(threshold_ratios)
    num_thresholds = len(threshold_ratios)

    adds_aps = np.zeros((num_classes + 1, num_thresholds), dtype=np.float64)
    pred_matches_all = [np.zeros((num_thresholds, 0), dtype=np.int32) for _ in range(num_classes)]
    gt_matches_all = [np.zeros((num_thresholds, 0), dtype=np.int32) for _ in range(num_classes)]
    pred_scores_all = [np.zeros((num_thresholds, 0), dtype=np.float64) for _ in range(num_classes)]
    matched_norm_errors = [[] for _ in range(num_classes)]

    for _, result in tqdm(enumerate(final_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        pred_class_ids = result['pred_class_ids'].astype(np.int32)
        pred_scores = np.asarray(result['pred_scores'])
        pred_RTs = np.array(result['pred_RTs'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):
            cls_gt_mask = gt_class_ids == cls_id
            cls_pred_mask = pred_class_ids == cls_id
            cls_gt_class_ids = gt_class_ids[cls_gt_mask] if len(gt_class_ids) else np.zeros(0, dtype=np.int32)
            cls_gt_RTs = gt_RTs[cls_gt_mask] if len(gt_class_ids) else np.zeros((0, 4, 4))
            cls_gt_scales = gt_scales[cls_gt_mask] if len(gt_class_ids) else np.zeros((0, 3))
            cls_pred_class_ids = pred_class_ids[cls_pred_mask] if len(pred_class_ids) else np.zeros(0, dtype=np.int32)
            cls_pred_scores = pred_scores[cls_pred_mask] if len(pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[cls_pred_mask] if len(pred_class_ids) else np.zeros((0, 4, 4))

            gt_match, pred_match, _, norm_errors, pred_indices = compute_adds_matches(
                cls_gt_class_ids,
                cls_gt_RTs,
                cls_gt_scales,
                cls_pred_class_ids,
                cls_pred_scores,
                cls_pred_RTs,
                synset_names,
                threshold_ratios=threshold_ratios,
                samples_per_axis=samples_per_axis,
            )
            if len(pred_indices):
                cls_pred_scores = cls_pred_scores[pred_indices]

            pred_matches_all[cls_id] = np.concatenate((pred_matches_all[cls_id], pred_match), axis=-1)
            gt_matches_all[cls_id] = np.concatenate((gt_matches_all[cls_id], gt_match), axis=-1)
            pred_scores_tile = np.tile(cls_pred_scores, (num_thresholds, 1))
            pred_scores_all[cls_id] = np.concatenate((pred_scores_all[cls_id], pred_scores_tile), axis=-1)

            if norm_errors.size:
                finite_errors = norm_errors[np.isfinite(norm_errors)]
                if finite_errors.size:
                    matched_norm_errors[cls_id].extend(finite_errors.tolist())

    for cls_id in range(1, num_classes):
        for t in range(num_thresholds):
            if gt_matches_all[cls_id].shape[1] == 0:
                adds_aps[cls_id, t] = 0.0
            else:
                adds_aps[cls_id, t] = compute_ap_from_matches_scores(
                    pred_matches_all[cls_id][t, :],
                    pred_scores_all[cls_id][t, :],
                    gt_matches_all[cls_id][t, :],
                )

    adds_aps[-1, :] = np.mean(adds_aps[1:-1, :], axis=0)

    if logger is not None:
        logger.warning('####### ADD-S result ###################')
        for t, threshold_ratio in enumerate(threshold_ratios):
            logger.warning('ADD-S AP at {:.0f}%% diameter: {:.1f}'.format(
                threshold_ratio * 100,
                adds_aps[-1, t] * 100,
            ))
        logger.warning('####### Per Class ADD-S result #########')
        for cls_id in range(1, len(synset_names)):
            logger.warning('category {}'.format(synset_names[cls_id]))
            for t, threshold_ratio in enumerate(threshold_ratios):
                logger.warning('ADD-S AP at {:.0f}%% diameter: {:.1f}'.format(
                    threshold_ratio * 100,
                    adds_aps[cls_id, t] * 100,
                ))
            if matched_norm_errors[cls_id]:
                logger.warning('mean ADD-S / diameter: {:.4f}'.format(
                    float(np.mean(matched_norm_errors[cls_id]))
                ))

    return adds_aps


def evaluate(path, logger=None, eval_protocol=DEFAULT_EVAL_PROTOCOL):
    """Run evaluate.

    Args:
        path: File path.
        logger: Logger instance.

    Returns:
        None.
    """
    synset_names = ['BG',
                    'bottle',
                    'bowl',
                    'camera',
                    'can',
                    'laptop',
                    'mug']

    result_pkl_list = glob.glob(os.path.join(path, 'results*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    print('image num: {}'.format(len(result_pkl_list)))

    final_results = []
    count = 0
    for pkl_path in result_pkl_list:

        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if not 'gt_handle_visibility' in result:
                result['gt_handle_visibility'] = np.ones_like(
                    result['gt_class_ids'])
                print('can\'t find gt_handle_visibility in the pkl.')
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])

        if type(result) is list:
            final_results += result
        elif type(result) is dict:
            final_results.append(result)
        else:
            assert False
        count+=1

    print("Compute independent mAP: ")
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    compute_independent_mAP(final_results, synset_names,
                            degree_thresholds=degree_thres_list,
                            shift_thresholds=shift_thres_list,
                            iou_3d_thresholds=iou_thres_list, logger=logger, log_dir=path,
                            eval_protocol=eval_protocol)
    print("Compute ADD-S mAP: ")
    compute_adds_mAP(final_results, synset_names, logger=logger)
