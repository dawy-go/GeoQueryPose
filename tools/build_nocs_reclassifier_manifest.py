#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.nocs_reclassifier import (  # noqa: E402
    CATEGORY_NAMES,
    RECLASSIFIER_CLASS_IDS,
    match_boxes_class_agnostic,
    sample_relative_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match REAL train detections to GT by class-agnostic 2D IoU and build "
            "the Bottle/Can/Mug reclassifier manifest."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reclassifier_data/real_train_bottle_can_mug.jsonl"),
    )
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument(
        "--target-class-ids",
        type=int,
        nargs="+",
        default=list(RECLASSIFIER_CLASS_IDS),
    )
    parser.add_argument(
        "--val-scenes",
        nargs="*",
        default=["scene_1"],
        help="Entire REAL train scenes reserved for validation.",
    )
    parser.add_argument(
        "--model-disjoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "exclude training rows whose physical model id appears in validation; "
            "enabled by default"
        ),
    )
    return parser.parse_args()


def _mask_source(result: dict, result_path: Path, data_root: Path, sample_rel: str) -> str:
    if result.get("pred_masks") is not None:
        return "pickle_pred_masks"
    indexed_path = result.get("pred_mask_path")
    if indexed_path:
        path = Path(str(indexed_path))
        if not path.is_absolute():
            path = result_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing indexed prediction mask: {path}")
        return "indexed_png"
    fallback = data_root / "Real" / f"{sample_rel}_mask_sam.png"
    if not fallback.is_file():
        raise FileNotFoundError(f"No prediction or fallback SAM mask for {sample_rel}")
    return "scene_sam_fallback"


def build_manifest(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if not 0.0 <= args.min_iou <= 1.0:
        raise ValueError("--min-iou must be in [0, 1]")
    target_ids = tuple(sorted(set(int(value) for value in args.target_class_ids)))
    invalid_ids = [value for value in target_ids if value <= 0 or value >= len(CATEGORY_NAMES)]
    if invalid_ids:
        raise ValueError(f"Invalid target class ids: {invalid_ids}")

    data_root = args.data_root.resolve()
    detections_dir = args.detections_dir.resolve()
    result_paths = sorted(detections_dir.glob("results_*.pkl"))
    if not result_paths:
        raise FileNotFoundError(f"No results_*.pkl files found in {detections_dir}")

    rows: list[dict] = []
    gt_counts: Counter[int] = Counter()
    matched_counts: Counter[int] = Counter()
    transition_counts: Counter[tuple[int, int]] = Counter()
    detector_sources: Counter[str] = Counter()
    skipped_files: list[dict[str, str]] = []

    for result_path in result_paths:
        try:
            with result_path.open("rb") as handle:
                result = pickle.load(handle)
            sample_rel = sample_relative_path(result_path, result)
            label_path = data_root / "Real" / f"{sample_rel}_label.pkl"
            with label_path.open("rb") as handle:
                gt = pickle.load(handle)

            pred_bboxes = np.asarray(result.get("pred_bboxes", []), dtype=np.float64).reshape(-1, 4)
            pred_class_ids = np.asarray(
                result.get("pred_class_ids", np.zeros(len(pred_bboxes))), dtype=np.int64
            ).reshape(-1)
            pred_scores = np.asarray(
                result.get("pred_scores", np.ones(len(pred_bboxes))), dtype=np.float64
            ).reshape(-1)
            gt_bboxes = np.asarray(gt.get("bboxes", []), dtype=np.float64).reshape(-1, 4)
            gt_class_ids = np.asarray(gt.get("class_ids", []), dtype=np.int64).reshape(-1)
            if len(pred_class_ids) != len(pred_bboxes):
                raise ValueError("pred_class_ids count does not match pred_bboxes")
            if len(gt_class_ids) != len(gt_bboxes):
                raise ValueError("GT class_ids count does not match bboxes")

            scene = sample_rel.split("/")[1]
            split = "val" if scene in set(args.val_scenes) else "train"
            for class_id in gt_class_ids:
                if int(class_id) in target_ids:
                    gt_counts[int(class_id)] += 1

            mask_source = _mask_source(result, result_path, data_root, sample_rel)
            model_list = list(gt.get("model_list", []))
            for pred_index, gt_index, iou in match_boxes_class_agnostic(
                pred_bboxes, gt_bboxes, min_iou=float(args.min_iou)
            ):
                gt_class_id = int(gt_class_ids[gt_index])
                if gt_class_id not in target_ids:
                    continue
                pred_class_id = int(pred_class_ids[pred_index])
                model_id = str(model_list[gt_index]) if gt_index < len(model_list) else ""
                rows.append(
                    {
                        "split": split,
                        "sample_rel": sample_rel,
                        "scene": scene,
                        "result_path": result_path.relative_to(detections_dir).as_posix(),
                        "mask_source": mask_source,
                        "pred_index": int(pred_index),
                        "pred_bbox_yxyx": pred_bboxes[pred_index].astype(float).tolist(),
                        "pred_class_id": pred_class_id,
                        "pred_class_name": (
                            CATEGORY_NAMES[pred_class_id]
                            if 0 <= pred_class_id < len(CATEGORY_NAMES)
                            else "unknown"
                        ),
                        "pred_score": (
                            float(pred_scores[pred_index])
                            if pred_index < len(pred_scores)
                            else -1.0
                        ),
                        "gt_index": int(gt_index),
                        "gt_bbox_yxyx": gt_bboxes[gt_index].astype(float).tolist(),
                        "gt_class_id": gt_class_id,
                        "gt_class_name": CATEGORY_NAMES[gt_class_id],
                        "gt_model_id": model_id,
                        "bbox_iou": float(iou),
                    }
                )
                matched_counts[gt_class_id] += 1
                transition_counts[(pred_class_id, gt_class_id)] += 1
            detector_sources[str(result.get("detector_source", "unknown"))] += 1
        except (FileNotFoundError, OSError, pickle.UnpicklingError, ValueError) as exc:
            skipped_files.append({"file": str(result_path), "reason": str(exc)})

    val_model_ids = {
        str(row["gt_model_id"])
        for row in rows
        if row["split"] == "val" and str(row["gt_model_id"])
    }
    excluded_model_overlap_count = 0
    if args.model_disjoint and val_model_ids:
        filtered_rows = []
        for row in rows:
            overlaps_validation = (
                row["split"] == "train"
                and bool(row["gt_model_id"])
                and str(row["gt_model_id"]) in val_model_ids
            )
            if overlaps_validation:
                excluded_model_overlap_count += 1
            else:
                filtered_rows.append(row)
        rows = filtered_rows
    split_counts = Counter(str(row["split"]) for row in rows)

    summary = {
        "schema_version": 1,
        "data_root": str(data_root),
        "detections_dir": str(detections_dir),
        "detector_sources": dict(detector_sources),
        "min_iou": float(args.min_iou),
        "target_class_ids": list(target_ids),
        "target_class_names": [CATEGORY_NAMES[index] for index in target_ids],
        "val_scenes": list(args.val_scenes),
        "model_disjoint": bool(args.model_disjoint),
        "validation_model_ids": sorted(val_model_ids),
        "excluded_model_overlap_count": excluded_model_overlap_count,
        "result_file_count": len(result_paths),
        "sample_count": len(rows),
        "split_counts": dict(split_counts),
        "gt_counts": {CATEGORY_NAMES[key]: value for key, value in sorted(gt_counts.items())},
        "matched_counts": {
            CATEGORY_NAMES[key]: value for key, value in sorted(matched_counts.items())
        },
        "matched_recall": {
            CATEGORY_NAMES[key]: matched_counts[key] / max(gt_counts[key], 1)
            for key in sorted(gt_counts)
        },
        "detector_to_gt": {
            f"{CATEGORY_NAMES[pred] if 0 <= pred < len(CATEGORY_NAMES) else 'unknown'}"
            f"->{CATEGORY_NAMES[gt]}": count
            for (pred, gt), count in sorted(transition_counts.items())
        },
        "skipped_file_count": len(skipped_files),
        "skipped_files": skipped_files[:100],
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    rows, summary = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"manifest: {args.output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
