from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw GT (red) and predicted (green) boxes from saved result PKLs."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("2d", "3d", "both"), default="3d")
    parser.add_argument(
        "--match-pred-to-gt",
        action="store_true",
        help="draw only predictions matched to ground truth",
    )
    return parser.parse_args()


def result_image_id(path: Path) -> int:
    prefix = "results_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"Unexpected result filename: {path.name}")
    return int(path.stem[len(prefix):])


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_dir / "bbox_gt_red_pred_green"
    )
    result_paths = sorted(result_dir.glob("results_*.pkl"))
    if not result_paths:
        raise FileNotFoundError(f"No results_*.pkl files found under: {result_dir}")

    from utils.solver import TestingSolver

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for index, result_path in enumerate(result_paths, start=1):
        with result_path.open("rb") as stream:
            result = pickle.load(stream)
        saved_path = TestingSolver._draw_box_to_image(
            None,
            result,
            str(output_dir),
            result_image_id(result_path),
            draw_2d=args.mode in ("2d", "both"),
            draw_3d=args.mode in ("3d", "both"),
            draw_gt=True,
            draw_pred=True,
            match_pred_to_gt=args.match_pred_to_gt,
        )
        if saved_path is not None:
            saved_count += 1
        if index == 1 or index % 100 == 0 or index == len(result_paths):
            print(
                f"[{index}/{len(result_paths)}] saved={saved_count} output={output_dir}",
                flush=True,
            )

    if saved_count != len(result_paths):
        raise RuntimeError(
            f"Expected {len(result_paths)} images, but saved {saved_count}"
        )


if __name__ == "__main__":
    main()
