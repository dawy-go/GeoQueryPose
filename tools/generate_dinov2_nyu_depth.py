"""Generate DINOv2-NYU metric depth maps for GeoQueryPose."""

from __future__ import annotations

import argparse
import gc
import itertools
import math
import os
from functools import partial
from pathlib import Path
import sys
import urllib.request

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data"
DEFAULT_DINOV2_ROOT = REPO_ROOT / "third_party" / "dinov2"
SOURCE_SPECS = {
    "camera-train": ("Camera", "train_list.txt"),
    "real-train": ("Real", "train_list.txt"),
}
BACKBONE_ARCHS = {"base": "vitb14", "large": "vitl14"}
OUTPUT_SUFFIXES = {
    "base": "_dinov2_depth_u16.png",
    "large": "_dinov2_large_depth_u16.png",
}
DEPTH_SCALE = 1000.0
MAX_DEPTH_M = np.iinfo(np.uint16).max / DEPTH_SCALE
DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple: int):
        super().__init__()
        self.multiple = int(multiple)

    def _pad(self, size: int):
        total = math.ceil(size / self.multiple) * self.multiple - size
        left = total // 2
        return left, total - left

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        pads = list(itertools.chain.from_iterable(self._pad(size) for size in tensor.shape[:1:-1]))
        return F.pad(tensor, pads)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--source",
        action="append",
        choices=("camera-train", "real-train", "real-test", "train", "all"),
        help="Repeat to select sources; default is all.",
    )
    parser.add_argument("--dinov2-root", type=Path, default=DEFAULT_DINOV2_ROOT)
    parser.add_argument("--backbone", choices=("base", "large", "both"), default="base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add_dinov2_to_path(root: Path) -> None:
    root = root.resolve()
    if not (root / "dinov2" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"DINOv2 source not found under {root}. Pass a DINOv2 checkout with "
            "--dinov2-root."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def create_depther(cfg, backbone):
    from dinov2.eval.depth.models import build_depther

    depther = build_depther(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    depther.backbone.forward = partial(
        backbone.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )
    if hasattr(backbone, "patch_size"):
        depther.backbone.register_forward_pre_hook(
            lambda _, inputs: CenterPadding(backbone.patch_size)(inputs[0])
        )
    return depther


def build_model(size: str, dinov2_root: Path, device: torch.device):
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    model_name = f"dinov2_{BACKBONE_ARCHS[size]}"
    backbone = torch.hub.load(
        repo_or_dir=str(dinov2_root.resolve()),
        model=model_name,
        source="local",
        pretrained=True,
    ).eval().to(device)
    config_url = f"{DINOV2_BASE_URL}/{model_name}/{model_name}_nyu_dpt_config.py"
    checkpoint_url = f"{DINOV2_BASE_URL}/{model_name}/{model_name}_nyu_dpt_head.pth"
    with urllib.request.urlopen(config_url) as response:
        cfg = Config.fromstring(response.read().decode("utf-8"), file_format=".py")
    model = create_depther(cfg, backbone)
    load_checkpoint(model, checkpoint_url, map_location="cpu")
    return model.eval().to(device)


def make_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        lambda tensor: 255.0 * tensor[:3],
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


def selected_sources(values):
    requested = values or ["all"]
    expanded = []
    for value in requested:
        if value == "all":
            candidates = ("camera-train", "real-train", "real-test")
        elif value == "train":
            candidates = ("camera-train", "real-train")
        else:
            candidates = (value,)
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def discover_images(dataset_root: Path, sources, limit):
    groups = {}
    for source in selected_sources(sources):
        if source == "real-test":
            images = sorted((dataset_root / "Real" / "test").glob("scene_*/*_color.png"))
        else:
            directory_name, list_name = SOURCE_SPECS[source]
            source_root = dataset_root / directory_name
            list_path = source_root / list_name
            if not list_path.is_file():
                raise FileNotFoundError(f"Missing split list: {list_path}")
            images = []
            for line in list_path.read_text(encoding="utf-8").splitlines():
                stem = line.strip().replace("\\", "/")
                if not stem or stem.startswith("#"):
                    continue
                suffix = stem if stem.endswith("_color.png") else stem + "_color.png"
                images.append(source_root / Path(suffix))
        if limit is not None:
            images = images[:limit]
        groups[source] = images
    return groups


def output_path(image_path: Path, backbone_size: str) -> Path:
    stem = image_path.name.removesuffix("_color.png")
    return image_path.with_name(stem + OUTPUT_SUFFIXES[backbone_size])


def valid_output(path: Path, height: int, width: int) -> bool:
    if not path.is_file():
        return False
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image is not None and image.shape == (height, width) and image.dtype == np.uint16


def quantize_metric_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError(f"Invalid DINOv2 depth shape/range: {depth.shape}")
    if float(depth.min()) < 0.0 or float(depth.max()) > MAX_DEPTH_M:
        raise ValueError(
            f"Depth range [{float(depth.min()):.6f}, {float(depth.max()):.6f}] m "
            f"cannot be represented as uint16 millimetres"
        )
    return np.rint(depth * DEPTH_SCALE).astype(np.uint16)


def save_atomic(path: Path, depth_mm: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.png")
    if not cv2.imwrite(str(temporary), depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise OSError(f"Failed to write {temporary}")
    os.replace(temporary, path)


def generate(size, images, dinov2_root, device, overwrite):
    model = build_model(size, dinov2_root, device)
    transform = make_transform()
    generated = skipped = 0
    try:
        for image_path in tqdm(images, desc=size, unit="image"):
            with Image.open(image_path) as file:
                image = file.convert("RGB")
                width, height = image.size
                target = output_path(image_path, size)
                if not overwrite and valid_output(target, height, width):
                    skipped += 1
                    continue
                batch = transform(image).unsqueeze(0).to(device, non_blocking=True)
            with torch.inference_mode():
                result = model.whole_inference(batch, img_meta=None, rescale=True)
            depth = result.detach().cpu().numpy().astype(np.float32, copy=False)
            if depth.shape[-2:] != (height, width):
                raise RuntimeError(f"Unexpected output {depth.shape} for {image_path}")
            save_atomic(target, quantize_metric_depth(depth))
            generated += 1
        return generated, skipped
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    groups = discover_images(args.dataset_root.resolve(), args.source, args.limit)
    for source, images in groups.items():
        print(f"{source}: {len(images)} images")
        if images:
            print(f"  first: {images[0]}")
    if args.dry_run:
        return

    unique_images = []
    seen = set()
    for images in groups.values():
        for image in images:
            key = str(image).casefold()
            if key not in seen:
                seen.add(key)
                unique_images.append(image)
    if not unique_images:
        raise FileNotFoundError("No input images discovered")

    add_dinov2_to_path(args.dinov2_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    sizes = ("base", "large") if args.backbone == "both" else (args.backbone,)
    for size in sizes:
        generated, skipped = generate(
            size,
            unique_images,
            args.dinov2_root,
            device,
            args.overwrite,
        )
        print(f"{size}: generated={generated}, skipped={skipped}")


if __name__ == "__main__":
    main()
