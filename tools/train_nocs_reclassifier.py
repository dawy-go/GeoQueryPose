#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import pickle
import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.nocs_reclassifier import (  # noqa: E402
    CATEGORY_NAMES,
    RECLASSIFIER_CLASS_IDS,
    RECLASSIFIER_CLASS_NAMES,
    NocsReclassifier,
    classification_metrics,
    confusion_matrix,
    detection_mask,
    prepare_reclassifier_input,
    read_sample_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NOCS Bottle/Can/Mug reclassifier.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("training_logs/nocs-reclassifier")
    )
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--padding-ratio", type=float, default=0.08)
    parser.add_argument("--background-keep", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@lru_cache(maxsize=128)
def _load_pickle(path: str) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


class ReclassifierDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        data_root: Path,
        detections_dir: Path,
        image_size: int,
        training: bool,
        padding_ratio: float,
        background_keep: float,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.detections_dir = detections_dir
        self.image_size = int(image_size)
        self.training = bool(training)
        self.padding_ratio = float(padding_ratio)
        self.background_keep = float(background_keep)
        self.class_to_index = {
            class_id: index for index, class_id in enumerate(RECLASSIFIER_CLASS_IDS)
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        sample_rel = str(row["sample_rel"])
        result_path = self.detections_dir / str(row["result_path"])
        result = _load_pickle(str(result_path))
        rgb = read_sample_rgb(self.data_root, sample_rel)
        mask = detection_mask(
            result=result,
            result_path=result_path,
            pred_index=int(row["pred_index"]),
            data_root=self.data_root,
            sample_rel=sample_rel,
            image_shape=rgb.shape[:2],
        )
        inputs = prepare_reclassifier_input(
            rgb=rgb,
            mask=mask,
            bbox_yxyx=row["pred_bbox_yxyx"],
            image_size=self.image_size,
            training=self.training,
            padding_ratio=self.padding_ratio,
            background_keep=self.background_keep,
        )
        class_id = int(row["gt_class_id"])
        if class_id not in self.class_to_index:
            raise ValueError(f"Manifest contains unsupported GT class id {class_id}")
        return inputs, self.class_to_index[class_id], int(row.get("pred_class_id", 0))


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_grad_scaler(device: torch.device, enabled: bool):
    """Use the AMP scaler API available in the installed PyTorch version."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            # Some transitional releases expose GradScaler without a device argument.
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return torch.amp.autocast(device_type=device.type, enabled=True)


def set_backbone_frozen(model: NocsReclassifier, frozen: bool) -> None:
    for parameter in model.network.parameters():
        parameter.requires_grad_(not frozen)
    if frozen:
        for parameter in model.network.conv1.parameters():
            parameter.requires_grad_(True)
        for parameter in model.network.fc.parameters():
            parameter.requires_grad_(True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> tuple[float, dict[str, object], dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    original_predictions: list[int] = []
    target_class_to_index = {
        class_id: index for index, class_id in enumerate(RECLASSIFIER_CLASS_IDS)
    }

    for inputs, target, original_class_id in loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast_context(device, amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, target)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = int(target.shape[0])
        total_loss += float(loss.detach()) * batch_size
        targets.extend(target.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        original_predictions.extend(
            [target_class_to_index.get(int(value), -1) for value in original_class_id.tolist()]
        )

    matrix = confusion_matrix(targets, predictions, len(RECLASSIFIER_CLASS_IDS))
    metrics = classification_metrics(matrix)
    valid_original = [index for index, value in enumerate(original_predictions) if value >= 0]
    baseline_matrix = confusion_matrix(
        [targets[index] for index in valid_original],
        [original_predictions[index] for index in valid_original],
        len(RECLASSIFIER_CLASS_IDS),
    )
    baseline = classification_metrics(baseline_matrix)
    baseline["excluded_unknown_or_non_target"] = len(targets) - len(valid_original)
    return total_loss / max(len(targets), 1), metrics, baseline


def metrics_with_names(metrics: dict[str, object]) -> dict[str, object]:
    output = dict(metrics)
    recalls = list(output.pop("per_class_recall"))
    output["per_class_recall"] = {
        name: float(recalls[index]) for index, name in enumerate(RECLASSIFIER_CLASS_NAMES)
    }
    return output


def save_checkpoint(
    path: Path,
    model: NocsReclassifier,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, object],
) -> None:
    checkpoint = {
        "schema_version": 1,
        "epoch": int(epoch),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "backbone": args.backbone,
        "dropout": float(args.dropout),
        "target_class_ids": list(RECLASSIFIER_CLASS_IDS),
        "target_class_names": list(RECLASSIFIER_CLASS_NAMES),
        "image_size": int(args.image_size),
        "padding_ratio": float(args.padding_ratio),
        "background_keep": float(args.background_keep),
        "validation_metrics": metrics,
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("epochs, batch-size, and image-size must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    rows = load_rows(args.manifest)
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    if not train_rows or not val_rows:
        raise ValueError(
            f"Manifest must contain train and val rows, got train={len(train_rows)} val={len(val_rows)}"
        )
    for split_name, split_rows in (("train", train_rows), ("val", val_rows)):
        split_class_ids = {int(row["gt_class_id"]) for row in split_rows}
        missing = sorted(set(RECLASSIFIER_CLASS_IDS) - split_class_ids)
        if missing:
            missing_names = [CATEGORY_NAMES[class_id] for class_id in missing]
            raise ValueError(
                f"{split_name} split has no matched samples for {missing_names}; "
                "change --val-scenes or lower the manifest IoU threshold only after auditing detections"
            )

    data_root = args.data_root.resolve()
    detections_dir = args.detections_dir.resolve()
    train_dataset = ReclassifierDataset(
        train_rows,
        data_root,
        detections_dir,
        args.image_size,
        True,
        args.padding_ratio,
        args.background_keep,
    )
    val_dataset = ReclassifierDataset(
        val_rows,
        data_root,
        detections_dir,
        args.image_size,
        False,
        args.padding_ratio,
        args.background_keep,
    )
    counts = Counter(int(row["gt_class_id"]) for row in train_rows)
    sample_weights = [1.0 / max(counts[int(row["gt_class_id"])], 1) for row in train_rows]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator
    )
    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_common)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_common)

    model = NocsReclassifier(
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)
    set_backbone_frozen(model, args.freeze_backbone_epochs > 0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.03
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = make_grad_scaler(device, amp_enabled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_balanced_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_frozen(model, False)
        train_loss, train_metrics, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_enabled
        )
        with torch.inference_mode():
            val_loss, val_metrics, detector_baseline = run_epoch(
                model, val_loader, criterion, device, None, scaler, amp_enabled
            )
        scheduler.step()
        named_train = metrics_with_names(train_metrics)
        named_val = metrics_with_names(val_metrics)
        named_baseline = metrics_with_names(detector_baseline)
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train": named_train,
            "val": named_val,
            "detector_baseline_on_val": named_baseline,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        score = float(val_metrics["balanced_accuracy"])
        if score > best_balanced_accuracy:
            best_balanced_accuracy = score
            stale_epochs = 0
            save_checkpoint(args.output_dir / "best.pth", model, epoch, args, named_val)
        else:
            stale_epochs += 1
        save_checkpoint(args.output_dir / "last.pth", model, epoch, args, named_val)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"early stopping after {stale_epochs} epochs without improvement")
            break

    print(f"best_balanced_accuracy={best_balanced_accuracy:.6f}")
    print(f"best_checkpoint={args.output_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
