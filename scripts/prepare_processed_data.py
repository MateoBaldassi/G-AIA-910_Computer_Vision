"""
scripts/prepare_processed_data.py
=================================
Build a pseudo-labeled dataset for CNN digit training from raw Sudoku screenshots.

Pipeline:
  1) Detect grid + 81 cell crops with YOLO
  2) Predict each cell digit with one or more OCR backends
  3) Keep only confident labels (consensus by default)
  4) Split into train/val/test ImageFolder layout:
       data/processed/{train,val,test}/{0..9}/*.png

This script is useful when fully manual labeling is not available.
Labels are pseudo-labels and may include noise.

Examples:
  python scripts/prepare_processed_data.py --difficulty easy --label-methods easyocr
  python scripts/prepare_processed_data.py --difficulty easy medium hard --label-methods easyocr,tesseract
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
import sys

import cv2

logger = logging.getLogger(__name__)


# Allow running this script directly from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="Root folder containing raw screenshots by difficulty")
    parser.add_argument("--output-dir", default="data/processed", help="Output root for ImageFolder splits")
    parser.add_argument("--model-path", default="models/yolo_sudoku.pt", help="YOLO weights used for cell extraction")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="YOLO detection confidence threshold")
    parser.add_argument(
        "--difficulty",
        nargs="+",
        default=["easy", "medium", "hard", "expert"],
        choices=["easy", "medium", "hard", "expert"],
        help="Difficulty subfolders from data/raw to include",
    )
    parser.add_argument(
        "--label-methods",
        default="easyocr",
        help="Comma-separated OCR methods from DigitRecognizer: easyocr,tesseract,template,cnn",
    )
    parser.add_argument(
        "--allow-disagreement",
        action="store_true",
        help="Disable strict consensus and keep majority-vote labels for non-agreeing cells",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-difficulty", type=int, default=0, help="0 = no limit")
    parser.add_argument("--keep-empty", action="store_true", default=True, help="Keep class 0 (empty) cells")
    parser.add_argument("--drop-empty", action="store_true", help="Drop class 0 cells")
    parser.add_argument("--clean-output", action="store_true", help="Delete output-dir before generating dataset")
    return parser.parse_args()


def _resolve_label(preds: list[int], consensus_only: bool) -> tuple[int | None, bool]:
    """
    Resolve a final class label from multiple predictions.

    Returns
    -------
    label, agreed
      label is None if no reliable label can be assigned.
    """
    valid = [p for p in preds if isinstance(p, int) and 0 <= p <= 9]
    if not valid:
        return None, False

    agreed = len(set(valid)) == 1
    if agreed:
        return valid[0], True

    if consensus_only:
        return None, False

    # Majority vote fallback
    counts = defaultdict(int)
    for p in valid:
        counts[p] += 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best, False


def _split_name(r: float, train_ratio: float, val_ratio: float) -> str:
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"


def _ensure_layout(output_dir: Path):
    for split in ("train", "val", "test"):
        for cls in range(10):
            (output_dir / split / str(cls)).mkdir(parents=True, exist_ok=True)


def _load_recognizers(methods: list[str]):
    from ocr.digit_recognizer import DigitRecognizer

    recs = []
    for method in methods:
        logger.info("Initializing recognizer: %s", method)
        recs.append((method, DigitRecognizer(method=method)._backend))
    return recs


def _looks_like_sudoku_detector(weights_path: Path) -> bool:
    """Return True if model class names contain both 'cell' and 'grid'."""
    try:
        from ultralytics import YOLO

        names = YOLO(str(weights_path)).names
        if not isinstance(names, dict):
            return False
        normalized = {str(v).strip().lower() for v in names.values()}
        return "cell" in normalized and "grid" in normalized
    except Exception:
        return False


def _resolve_model_path(requested_path: str) -> Path:
    """Pick a usable Sudoku detector weights path, falling back to known run outputs."""
    candidates = [
        Path(requested_path),
        PROJECT_ROOT / "runs" / "detect" / "sudoku" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "sudoku" / "weights" / "best.pt",
    ]

    for p in candidates:
        if p.exists() and _looks_like_sudoku_detector(p):
            if str(p) != requested_path:
                logger.warning("Using fallback Sudoku weights: %s", p)
            return p

    existing = [str(p) for p in candidates if p.exists()]
    raise FileNotFoundError(
        "Could not find Sudoku-trained YOLO weights with classes {cell, grid}. "
        f"Checked: {existing or [str(c) for c in candidates]}"
    )


def build_dataset(args: argparse.Namespace):
    from detection.grid_detector import GridDetector

    if args.drop_empty:
        args.keep_empty = False

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_layout(out_dir)

    methods = [m.strip() for m in args.label_methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("No label methods provided")

    consensus_only = not args.allow_disagreement

    rng = random.Random(args.seed)
    resolved_model = _resolve_model_path(args.model_path)
    detector = GridDetector(model_path=str(resolved_model), conf_threshold=args.conf_threshold)
    recognizers = _load_recognizers(methods)

    stats = {
        "screenshots": 0,
        "cells_seen": 0,
        "cells_saved": 0,
        "cells_skipped_no_label": 0,
        "cells_skipped_empty": 0,
        "consensus_kept": 0,
        "majority_kept": 0,
    }
    class_counts = defaultdict(int)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8") as mf:
        mf.write("split,label,source,row,col,filename,agreed\n")

        for difficulty in args.difficulty:
            img_paths = sorted((raw_dir / difficulty).glob("*.png"))
            if args.max_images_per_difficulty > 0:
                img_paths = img_paths[: args.max_images_per_difficulty]

            logger.info("Processing %d screenshots from %s", len(img_paths), difficulty)

            for img_path in img_paths:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning("Failed to read image: %s", img_path)
                    continue

                stats["screenshots"] += 1

                try:
                    _, cell_grid = detector.detect(img)
                except Exception as e:
                    logger.warning("Detection failed for %s: %s", img_path.name, e)
                    continue

                if len(cell_grid) != 9 or any(len(row) != 9 for row in cell_grid):
                    logger.warning("Unexpected cell grid shape for %s", img_path.name)
                    continue

                for r, row in enumerate(cell_grid):
                    for c, cell_img in enumerate(row):
                        stats["cells_seen"] += 1

                        preds = []
                        for _, backend in recognizers:
                            try:
                                preds.append(int(backend.predict_cell(cell_img)))
                            except Exception:
                                # Ignore backend-specific failures for this one cell.
                                continue

                        label, agreed = _resolve_label(preds, consensus_only=consensus_only)
                        if label is None:
                            stats["cells_skipped_no_label"] += 1
                            continue
                        if label == 0 and not args.keep_empty:
                            stats["cells_skipped_empty"] += 1
                            continue

                        split = _split_name(rng.random(), args.train_ratio, args.val_ratio)
                        out_name = f"{difficulty}_{img_path.stem}_r{r}c{c}.png"
                        out_path = out_dir / split / str(label) / out_name

                        cv2.imwrite(str(out_path), cell_img)
                        mf.write(f"{split},{label},{img_path.as_posix()},{r},{c},{out_name},{int(agreed)}\n")

                        stats["cells_saved"] += 1
                        class_counts[label] += 1
                        if agreed:
                            stats["consensus_kept"] += 1
                        else:
                            stats["majority_kept"] += 1

    logger.info("Done. Processed dataset created at %s", out_dir)
    logger.info(
        "Screenshots=%d | Cells seen=%d | Saved=%d | Skipped(no label)=%d | Skipped(empty)=%d",
        stats["screenshots"],
        stats["cells_seen"],
        stats["cells_saved"],
        stats["cells_skipped_no_label"],
        stats["cells_skipped_empty"],
    )
    logger.info(
        "Saved labels by class: %s",
        ", ".join(f"{d}:{class_counts[d]}" for d in range(10) if class_counts[d] > 0) or "none",
    )
    logger.info("Manifest: %s", manifest_path)


def main():
    args = _parse_args()
    build_dataset(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
