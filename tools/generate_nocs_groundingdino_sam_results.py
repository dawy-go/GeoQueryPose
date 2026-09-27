#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.nocs_reclassifier import (  # noqa: E402
    CATEGORY_NAMES,
    detector_label_to_class_id,
)


REAL275_GT_KEYS = (
    "gt_class_ids",
    "gt_bboxes",
    "gt_RTs",
    "gt_scales",
    "gt_handle_visibility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NOCS-style REAL results with GroundingDINO and SAM."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-file", default="Real/train_list.txt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/segmentation_results/REAL_train_groundingdino_sam"),
    )
    parser.add_argument(
        "--grounding-model", default="IDEA-Research/grounding-dino-base"
    )
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_NAMES[1:]))
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=16)
    parser.add_argument(
        "--sam-box-batch-size",
        type=int,
        default=4,
        help="maximum boxes sent through SAM at once; lower this when cuDNN runs out of plans",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help=(
            "optional directory of same-named REAL275 result PKLs; copies the GT "
            "fields required by test.py without copying its detector predictions"
        ),
    )
    parser.add_argument(
        "--require-gt-metadata",
        action="store_true",
        help="fail an image when its same-named metadata PKL or required GT fields are missing",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable cuDNN for environments that consistently report no execution engine",
    )
    return parser.parse_args()


def load_samples(data_root: Path, split_file: str) -> list[str]:
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    with split_path.open("r", encoding="utf-8") as handle:
        return [line.strip().replace("\\", "/") for line in handle if line.strip()]


def load_models(args: argparse.Namespace):
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        SamModel,
        SamProcessor,
    )

    grounding_processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model
    ).to(args.device)
    grounding_model.eval()
    sam_processor = SamProcessor.from_pretrained(args.sam_model)
    sam_model = SamModel.from_pretrained(args.sam_model).to(args.device)
    sam_model.eval()
    return grounding_processor, grounding_model, sam_processor, sam_model


def post_process_grounding_dino(
    processor,
    outputs,
    input_ids: torch.Tensor,
    target_sizes: torch.Tensor,
    box_threshold: float,
    text_threshold: float,
):
    """Call both the old (`box_threshold`) and new (`threshold`) HF APIs."""
    method = processor.post_process_grounded_object_detection
    parameters = inspect.signature(method).parameters
    kwargs = {"outputs": outputs}
    if "input_ids" in parameters:
        kwargs["input_ids"] = input_ids
    if "target_sizes" in parameters:
        kwargs["target_sizes"] = target_sizes
    if "text_threshold" in parameters:
        kwargs["text_threshold"] = float(text_threshold)
    if "threshold" in parameters:
        kwargs["threshold"] = float(box_threshold)
    elif "box_threshold" in parameters:
        kwargs["box_threshold"] = float(box_threshold)
    else:
        raise TypeError(
            "Unsupported GroundingDINO post-process API: expected threshold or box_threshold, "
            f"got {tuple(parameters)}"
        )
    return method(**kwargs)


def detect(
    image_rgb: np.ndarray,
    categories: list[str],
    processor,
    model,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prompt = " . ".join(categories) + " ."
    inputs = processor(
        images=Image.fromarray(image_rgb), text=prompt, return_tensors="pt"
    ).to(args.device)
    amp_enabled = bool(args.amp and str(args.device).startswith("cuda"))
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
    ):
        outputs = model(**inputs)
    target_sizes = torch.tensor(
        [image_rgb.shape[:2]], device=inputs["pixel_values"].device
    )
    results = post_process_grounding_dino(
        processor=processor,
        outputs=outputs,
        input_ids=inputs["input_ids"],
        target_sizes=target_sizes,
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
    )
    if not results or len(results[0]["boxes"]) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), []

    boxes = results[0]["boxes"].detach().cpu()
    scores = results[0]["scores"].detach().cpu()
    raw_labels = results[0].get("text_labels", results[0].get("labels"))
    if raw_labels is None:
        raise KeyError(
            f"GroundingDINO result has neither text_labels nor labels: {tuple(results[0])}"
        )
    labels = list(raw_labels)
    keep = nms(boxes, scores, float(args.nms_threshold))
    if args.max_detections > 0:
        keep = keep[: args.max_detections]
    boxes = boxes[keep].numpy()
    scores = scores[keep].numpy()
    labels = [labels[index] for index in keep.tolist()]

    valid_indices = [
        index for index, label in enumerate(labels) if detector_label_to_class_id(label) > 0
    ]
    if not valid_indices:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), []
    return boxes[valid_indices], scores[valid_indices], [labels[index] for index in valid_indices]


def segment(
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    processor,
    model,
    device: str,
    box_batch_size: int,
    amp_enabled: bool,
) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.zeros((0, *image_rgb.shape[:2]), dtype=bool)
    all_masks = []
    for offset in range(0, len(boxes_xyxy), box_batch_size):
        box_chunk = boxes_xyxy[offset : offset + box_batch_size]
        inputs = processor(
            images=image_rgb,
            input_boxes=[[box.tolist() for box in box_chunk]],
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            outputs = model(**inputs, multimask_output=False)
        masks = processor.image_processor.post_process_masks(
            # SAM runs under CUDA AMP, while the processor resizes masks on CPU.
            # PyTorch 2.1 CPU interpolation has no float16 implementation.
            outputs.pred_masks.detach().float().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        all_masks.append(masks.squeeze(1).numpy() > 0)
        del inputs, outputs, masks
    return np.concatenate(all_masks, axis=0).astype(bool, copy=False)


def clear_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cudnn_engine_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "unable to find an engine" in message or "cudnn_status" in message


def boxes_xyxy_to_yxyx(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    output = []
    for x1, y1, x2, y2 in boxes:
        x1 = int(np.floor(np.clip(x1, 0, max(width - 1, 0))))
        y1 = int(np.floor(np.clip(y1, 0, max(height - 1, 0))))
        x2 = int(np.ceil(np.clip(x2, x1 + 1, width)))
        y2 = int(np.ceil(np.clip(y2, y1 + 1, height)))
        output.append([y1, x1, y2, x2])
    return np.asarray(output, dtype=np.int32).reshape(-1, 4)


def save_indexed_masks(path: Path, masks: np.ndarray) -> np.ndarray:
    if len(masks) > 255:
        raise ValueError("Indexed PNG mask supports at most 255 detections")
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shaped [N,H,W], got {masks.shape}")
    height, width = masks.shape[1:]
    indexed = np.zeros((height, width), dtype=np.uint8)
    # Earlier detections have higher confidence and win overlap pixels.
    for index in reversed(range(len(masks))):
        indexed[masks[index]] = index + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), indexed):
        raise OSError(f"Failed to write indexed mask: {path}")
    return np.arange(1, len(masks) + 1, dtype=np.uint8)


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


def merge_gt_metadata(
    payload: dict,
    metadata_dir: Path | None,
    result_name: str,
    required: bool = False,
) -> bool:
    if metadata_dir is None:
        if required:
            raise ValueError("--require-gt-metadata requires --metadata-dir")
        return False

    metadata_path = metadata_dir / result_name
    if not metadata_path.is_file():
        if required:
            raise FileNotFoundError(f"Missing REAL275 metadata result: {metadata_path}")
        return False
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    missing_keys = [key for key in REAL275_GT_KEYS if key not in metadata]
    if missing_keys:
        raise KeyError(
            f"REAL275 metadata result {metadata_path} is missing fields: {missing_keys}"
        )
    payload.update({key: metadata[key] for key in REAL275_GT_KEYS})
    return True


def main() -> None:
    args = parse_args()
    if args.image_offset < 0 or args.max_images < 0:
        raise ValueError("--image-offset and --max-images must be non-negative")
    if args.sam_box_batch_size <= 0:
        raise ValueError("--sam-box-batch-size must be positive")
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    metadata_dir = args.metadata_dir.resolve() if args.metadata_dir else None
    if args.require_gt_metadata and metadata_dir is None:
        raise ValueError("--require-gt-metadata requires --metadata-dir")
    if metadata_dir is not None and not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory does not exist: {metadata_dir}")
    mask_dir = output_dir / "masks"
    sample_paths = load_samples(data_root, args.split_file)[args.image_offset :]
    if args.max_images:
        sample_paths = sample_paths[: args.max_images]
    if args.disable_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for detection generation")

    models = None
    completed = 0
    skipped = 0
    failed = 0
    metadata_merged = 0
    for sequence_index, sample_rel_with_root in enumerate(sample_paths, start=1):
        sample_rel_with_root = sample_rel_with_root.removeprefix("./")
        if sample_rel_with_root.lower().startswith("real/"):
            sample_rel = sample_rel_with_root[len("Real/") :]
        else:
            sample_rel = sample_rel_with_root
        result_stem = "results_" + sample_rel.replace("/", "_")
        result_path = output_dir / f"{result_stem}.pkl"
        mask_path = mask_dir / f"{result_stem}.png"
        if result_path.is_file() and mask_path.is_file() and not args.overwrite:
            try:
                if metadata_dir is not None:
                    with result_path.open("rb") as handle:
                        existing_payload = pickle.load(handle)
                    if merge_gt_metadata(
                        existing_payload,
                        metadata_dir,
                        result_path.name,
                        required=args.require_gt_metadata,
                    ):
                        atomic_pickle_dump(existing_payload, result_path)
                        metadata_merged += 1
                skipped += 1
                continue
            except Exception as exc:
                failed += 1
                print(
                    f"[{sequence_index}/{len(sample_paths)}] FAIL "
                    f"{sample_rel} stage=metadata: {exc}"
                )
                continue
        try:
            active_stage = "load_rgb"
            image_path = data_root / "Real" / f"{sample_rel}_color.png"
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Failed to read RGB image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            active_stage = "groundingdino"
            if models is None:
                models = load_models(args)
            processor, detector, sam_processor, sam_model = models
            try:
                boxes_xyxy, scores, labels = detect(
                    image_rgb, list(args.categories), processor, detector, args
                )
            except RuntimeError as exc:
                if not is_cudnn_engine_error(exc):
                    raise
                print(
                    f"[{sequence_index}/{len(sample_paths)}] RETRY {sample_rel}: "
                    "detector without cuDNN; cuDNN disabled for remaining images"
                )
                clear_cuda_state()
                torch.backends.cudnn.enabled = False
                args.disable_cudnn = True
                boxes_xyxy, scores, labels = detect(
                    image_rgb, list(args.categories), processor, detector, args
                )

            active_stage = "sam"
            try:
                masks = segment(
                    image_rgb,
                    boxes_xyxy,
                    sam_processor,
                    sam_model,
                    args.device,
                    args.sam_box_batch_size,
                    bool(args.amp and str(args.device).startswith("cuda")),
                )
            except RuntimeError as exc:
                if not is_cudnn_engine_error(exc) or args.sam_box_batch_size == 1:
                    raise
                print(
                    f"[{sequence_index}/{len(sample_paths)}] RETRY {sample_rel}: "
                    "SAM one box at a time without cuDNN"
                )
                clear_cuda_state()
                torch.backends.cudnn.enabled = False
                args.disable_cudnn = True
                masks = segment(
                    image_rgb,
                    boxes_xyxy,
                    sam_processor,
                    sam_model,
                    args.device,
                    1,
                    False,
                )
            active_stage = "save"
            mask_labels = save_indexed_masks(mask_path, masks)
            pred_class_ids = np.asarray(
                [detector_label_to_class_id(label) for label in labels], dtype=np.int64
            )
            payload = {
                "schema_version": 1,
                "image_path": f"Real/{sample_rel}",
                "detector_source": "groundingdino_sam_fallback",
                "detector_model": args.grounding_model,
                "segmenter_model": args.sam_model,
                "pred_class_ids": pred_class_ids,
                "pred_bboxes": boxes_xyxy_to_yxyx(boxes_xyxy, image_rgb.shape[:2]),
                "pred_scores": np.asarray(scores, dtype=np.float32),
                "pred_labels": labels,
                "pred_mask_path": mask_path.relative_to(output_dir).as_posix(),
                "pred_mask_labels": mask_labels,
                "bbox_format": "yxyx",
                "box_threshold": float(args.box_threshold),
                "text_threshold": float(args.text_threshold),
                "nms_threshold": float(args.nms_threshold),
            }
            if merge_gt_metadata(
                payload,
                metadata_dir,
                result_path.name,
                required=args.require_gt_metadata,
            ):
                metadata_merged += 1
            atomic_pickle_dump(payload, result_path)
            completed += 1
            print(
                f"[{sequence_index}/{len(sample_paths)}] {sample_rel}: "
                f"detections={len(pred_class_ids)}"
            )
        except Exception as exc:
            failed += 1
            print(
                f"[{sequence_index}/{len(sample_paths)}] FAIL "
                f"{sample_rel} stage={active_stage}: {exc}"
            )
            clear_cuda_state()
    print(
        f"completed={completed} skipped={skipped} failed={failed} "
        f"metadata_merged={metadata_merged} output={output_dir}"
    )
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
