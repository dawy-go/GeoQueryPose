#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.nocs_reclassifier import (  # noqa: E402
    CATEGORY_NAMES,
    NocsReclassifier,
    detection_mask,
    prepare_reclassifier_input,
    read_sample_rgb,
    sample_relative_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained Bottle/Can/Mug reclassifier to NOCS result PKLs."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument(
        "--candidate-class-ids",
        type=int,
        nargs="+",
        default=None,
        help="Classes routed to the reclassifier; defaults to checkpoint target classes.",
    )
    parser.add_argument(
        "--allowed-transitions",
        nargs="+",
        default=None,
        metavar="FROM:TO",
        help=(
            "optional whitelist for class-changing predictions, using names or IDs "
            "(for example bottle:can or 1:4); same-class predictions remain allowed"
        ),
    )
    parser.add_argument(
        "--score-mode",
        choices=["preserve", "multiply", "reclassifier"],
        default="preserve",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _class_token_to_id(token: str) -> int:
    normalized = str(token).strip().lower()
    name_to_id = {name.lower(): index for index, name in enumerate(CATEGORY_NAMES)}
    if normalized in name_to_id:
        class_id = name_to_id[normalized]
    else:
        try:
            class_id = int(normalized)
        except ValueError as exc:
            raise ValueError(f"Unknown NOCS class in transition: {token!r}") from exc
    if class_id <= 0 or class_id >= len(CATEGORY_NAMES):
        raise ValueError(
            f"Transition class ID must be in [1, {len(CATEGORY_NAMES) - 1}], got {class_id}"
        )
    return class_id


def parse_allowed_transitions(values: list[str] | None) -> set[tuple[int, int]] | None:
    if values is None:
        return None
    transitions: set[tuple[int, int]] = set()
    for value in values:
        separator = "->" if "->" in value else ":"
        parts = value.split(separator)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(
                f"Invalid transition {value!r}; expected FROM:TO, for example bottle:can"
            )
        transitions.add((_class_token_to_id(parts[0]), _class_token_to_id(parts[1])))
    return transitions


def transition_names(transitions: set[tuple[int, int]] | None) -> list[str] | None:
    if transitions is None:
        return None
    return [
        f"{CATEGORY_NAMES[before]}->{CATEGORY_NAMES[after]}"
        for before, after in sorted(transitions)
    ]


def atomic_pickle_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rebase_pred_mask_path(
    result: dict,
    source_result_path: Path,
    output_result_path: Path,
) -> dict:
    """Keep a relative indexed-mask reference valid after moving its result PKL."""
    mask_value = result.get("pred_mask_path")
    if not mask_value:
        return result

    source_mask_path = Path(str(mask_value))
    if not source_mask_path.is_absolute():
        source_mask_path = source_result_path.parent / source_mask_path
    source_mask_path = source_mask_path.resolve()
    try:
        rebased = Path(os.path.relpath(source_mask_path, output_result_path.parent.resolve()))
    except ValueError:
        # Windows paths on different drives cannot be expressed relatively.
        rebased = source_mask_path
    result["pred_mask_path"] = rebased.as_posix()
    return result


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_ids = tuple(int(value) for value in checkpoint["target_class_ids"])
    model = NocsReclassifier(
        backbone=str(checkpoint.get("backbone", "resnet18")),
        num_classes=len(target_ids),
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.20)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    preprocessing = {
        "image_size": int(checkpoint.get("image_size", 224)),
        "padding_ratio": float(checkpoint.get("padding_ratio", 0.08)),
        "background_keep": float(checkpoint.get("background_keep", 0.20)),
    }
    return model, target_ids, preprocessing, checkpoint


def apply_to_result(
    result: dict,
    result_path: Path,
    data_root: Path,
    model: NocsReclassifier,
    target_ids: tuple[int, ...],
    preprocessing: dict[str, float | int],
    candidate_ids: set[int],
    device: torch.device,
    batch_size: int,
    min_confidence: float,
    min_margin: float,
    score_mode: str,
    allowed_transitions: set[tuple[int, int]] | None = None,
) -> tuple[dict, Counter]:
    original_ids = np.asarray(result.get("pred_class_ids", []), dtype=np.int64).reshape(-1)
    bboxes = np.asarray(result.get("pred_bboxes", []), dtype=np.float64).reshape(-1, 4)
    if len(original_ids) != len(bboxes):
        raise ValueError("pred_class_ids count does not match pred_bboxes")
    output = dict(result)
    corrected_ids = original_ids.copy()
    scores = np.asarray(result.get("pred_scores", np.ones(len(bboxes))), dtype=np.float32).copy()
    eligible = np.asarray([int(value) in candidate_ids for value in original_ids], dtype=bool)
    proposed_ids = original_ids.copy()
    probabilities = np.full((len(bboxes), len(target_ids)), np.nan, dtype=np.float32)
    confidence = np.zeros(len(bboxes), dtype=np.float32)
    margin = np.zeros(len(bboxes), dtype=np.float32)
    threshold_passed = np.zeros(len(bboxes), dtype=bool)
    transition_allowed = np.zeros(len(bboxes), dtype=bool)
    accepted = np.zeros(len(bboxes), dtype=bool)

    if np.any(eligible):
        sample_rel = sample_relative_path(result_path, result)
        rgb = read_sample_rgb(data_root, sample_rel)
        candidate_indices = np.flatnonzero(eligible).tolist()
        tensors = []
        for pred_index in candidate_indices:
            mask = detection_mask(
                result=result,
                result_path=result_path,
                pred_index=pred_index,
                data_root=data_root,
                sample_rel=sample_rel,
                image_shape=rgb.shape[:2],
            )
            tensors.append(
                prepare_reclassifier_input(
                    rgb=rgb,
                    mask=mask,
                    bbox_yxyx=bboxes[pred_index],
                    image_size=int(preprocessing["image_size"]),
                    training=False,
                    padding_ratio=float(preprocessing["padding_ratio"]),
                    background_keep=float(preprocessing["background_keep"]),
                )
            )
        for offset in range(0, len(tensors), batch_size):
            batch_indices = candidate_indices[offset : offset + batch_size]
            batch = torch.stack(tensors[offset : offset + batch_size]).to(device)
            with torch.inference_mode():
                batch_probabilities = model(batch).softmax(dim=1).cpu().numpy()
            for local_index, pred_index in enumerate(batch_indices):
                probs = batch_probabilities[local_index]
                order = np.argsort(probs)[::-1]
                predicted_class_id = target_ids[int(order[0])]
                probabilities[pred_index] = probs
                proposed_ids[pred_index] = predicted_class_id
                confidence[pred_index] = float(probs[order[0]])
                margin[pred_index] = float(
                    probs[order[0]] - probs[order[1]] if len(order) > 1 else probs[order[0]]
                )
                threshold_passed[pred_index] = bool(
                    confidence[pred_index] >= min_confidence
                    and margin[pred_index] >= min_margin
                )
                transition_allowed[pred_index] = bool(
                    predicted_class_id == int(original_ids[pred_index])
                    or allowed_transitions is None
                    or (int(original_ids[pred_index]), predicted_class_id)
                    in allowed_transitions
                )
                accepted[pred_index] = bool(
                    threshold_passed[pred_index] and transition_allowed[pred_index]
                )
                if accepted[pred_index]:
                    corrected_ids[pred_index] = predicted_class_id
                    if pred_index < len(scores):
                        if score_mode == "multiply":
                            scores[pred_index] *= confidence[pred_index]
                        elif score_mode == "reclassifier":
                            scores[pred_index] = confidence[pred_index]

    changed = corrected_ids != original_ids
    output["pred_class_ids"] = corrected_ids
    output["pred_scores"] = scores
    output["pred_class_ids_before_reclassification"] = original_ids
    output["pred_reclassifier_proposed_class_ids"] = proposed_ids
    output["pred_reclassifier_probabilities"] = probabilities
    output["pred_reclassifier_confidence"] = confidence
    output["pred_reclassifier_margin"] = margin
    output["pred_reclassifier_eligible"] = eligible
    output["pred_reclassifier_threshold_passed"] = threshold_passed
    output["pred_reclassifier_transition_allowed"] = transition_allowed
    output["pred_reclassifier_accepted"] = accepted
    output["pred_reclassifier_applied"] = changed
    output["pred_reclassifier_target_class_ids"] = np.asarray(target_ids, dtype=np.int64)

    transitions = Counter()
    for before, after in zip(original_ids[changed], corrected_ids[changed]):
        transitions[(int(before), int(after))] += 1
    transitions[("eligible", "count")] = int(eligible.sum())
    transitions[("threshold_passed", "count")] = int(threshold_passed.sum())
    transitions[("policy_rejected", "count")] = int(
        np.logical_and(threshold_passed, ~transition_allowed).sum()
    )
    transitions[("accepted", "count")] = int(accepted.sum())
    transitions[("changed", "count")] = int(changed.sum())
    return output, transitions


def main() -> None:
    args = parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("--output-dir must differ from --input-dir")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.min_confidence <= 1.0 or not 0.0 <= args.min_margin <= 1.0:
        raise ValueError("confidence and margin thresholds must be in [0, 1]")
    device = torch.device(args.device)
    model, target_ids, preprocessing, checkpoint = load_model(args.checkpoint, device)
    candidate_ids = set(args.candidate_class_ids or target_ids)
    allowed_transitions = parse_allowed_transitions(args.allowed_transitions)
    if allowed_transitions is not None:
        invalid_sources = sorted({before for before, _ in allowed_transitions} - candidate_ids)
        invalid_targets = sorted({after for _, after in allowed_transitions} - set(target_ids))
        if invalid_sources:
            raise ValueError(
                "Allowed transition sources are not candidate classes: "
                f"{invalid_sources}"
            )
        if invalid_targets:
            raise ValueError(
                "Allowed transition targets are not checkpoint classes: "
                f"{invalid_targets}"
            )
    input_paths = sorted(args.input_dir.resolve().glob("results_*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"No results_*.pkl files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    totals: Counter = Counter()
    completed = 0
    skipped = 0
    for result_path in input_paths:
        output_path = args.output_dir / result_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        with result_path.open("rb") as handle:
            result = pickle.load(handle)
        output, transitions = apply_to_result(
            result=result,
            result_path=result_path,
            data_root=args.data_root.resolve(),
            model=model,
            target_ids=target_ids,
            preprocessing=preprocessing,
            candidate_ids=candidate_ids,
            device=device,
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            score_mode=args.score_mode,
            allowed_transitions=allowed_transitions,
        )
        output["reclassifier_checkpoint"] = str(args.checkpoint.resolve())
        output["reclassifier_min_confidence"] = float(args.min_confidence)
        output["reclassifier_min_margin"] = float(args.min_margin)
        output["reclassifier_score_mode"] = args.score_mode
        output["reclassifier_allowed_transitions"] = transition_names(allowed_transitions)
        rebase_pred_mask_path(output, result_path, output_path)
        atomic_pickle_dump(output, output_path)
        totals.update(transitions)
        completed += 1

    named_transitions = {}
    for (before, after), count in totals.items():
        if isinstance(before, int):
            before_name = CATEGORY_NAMES[before] if 0 <= before < len(CATEGORY_NAMES) else str(before)
            after_name = CATEGORY_NAMES[after] if 0 <= after < len(CATEGORY_NAMES) else str(after)
            key = f"{before_name}->{after_name}"
        else:
            key = str(before)
        named_transitions[key] = int(count)
    summary = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "result_file_count": len(input_paths),
        "completed": completed,
        "skipped": skipped,
        "target_class_ids": list(target_ids),
        "candidate_class_ids": sorted(candidate_ids),
        "allowed_transitions": transition_names(allowed_transitions),
        "min_confidence": float(args.min_confidence),
        "min_margin": float(args.min_margin),
        "score_mode": args.score_mode,
        "transitions": named_transitions,
    }
    summary_path = args.output_dir / "reclassifier_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
