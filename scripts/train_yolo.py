"""
scripts/train_yolo.py
======================
Fine-tune YOLOv11 on Sudoku grid + cell detection.

Dataset structure (after annotation with Roboflow / CVAT):
    data/annotated/
        images/
            train/  ← screenshots of sudoku.com
            val/
        labels/
            train/  ← YOLO format .txt files
            val/
        data.yaml

Classes:
    0: cell   — individual cell (81 per image)
    1: grid   — full 9×9 bounding box

Usage:
    python scripts/train_yolo.py --epochs 50 --imgsz 640
    python scripts/train_yolo.py --resume runs/detect/sudoku/weights/last.pt
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_data_yaml(annotated_dir: str = "data/annotated") -> str:
    """Write the YOLO data.yaml if it doesn't already exist."""
    yaml_path = Path(annotated_dir) / "data.yaml"
    if yaml_path.exists():
        return str(yaml_path)

    content = f"""\
path: {Path(annotated_dir).resolve()}
train: images/train
val: images/val

nc: 2
names:
    0: cell
    1: grid
"""
    yaml_path.write_text(content)
    logger.info(f"Created {yaml_path}")
    return str(yaml_path)


def train(
    model_variant: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 10,
    resume: str = None,
    annotated_dir: str = "data/annotated",
    output_dir: str = "models",
    device: str | None = None,
):
    from ultralytics import YOLO

    data_yaml = create_data_yaml(annotated_dir)

    if resume:
        logger.info(f"Resuming from {resume}")
        model = YOLO(resume)
    else:
        logger.info(f"Starting fine-tune from {model_variant}")
        model = YOLO(model_variant)

    selected_device = _select_device(device)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project="runs/detect",
        name="sudoku",
        exist_ok=True,
        device=selected_device,
        verbose=True,
        # Augmentations suited for screenshot data
        hsv_h=0.01,
        hsv_s=0.1,
        hsv_v=0.2,
        translate=0.05,
        scale=0.1,
        fliplr=0.0,    # Don't flip — sudoku digits are orientation-sensitive
        flipud=0.0,
        mosaic=0.5,
    )

    # Export best weights
    best_weights = Path("runs/detect/sudoku/weights/best.pt")
    out = Path(output_dir) / "yolo_sudoku.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, out)
        logger.info(f"Best model saved to {out}")

    # Print final metrics
    metrics = model.val()
    logger.info(f"\n=== Validation Metrics ===")
    logger.info(f"mAP@0.5       : {metrics.box.map50:.4f}")
    logger.info(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
    logger.info(f"Precision     : {metrics.box.mp:.4f}")
    logger.info(f"Recall        : {metrics.box.mr:.4f}")

    return results


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _select_device(requested_device: str | None = None) -> str:
    if requested_device:
        logger.info(f"Using requested device: {requested_device}")
        return requested_device

    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "0"

        if torch.version.cuda is None:
            logger.warning(
                "CUDA unavailable because the installed PyTorch build is CPU-only. "
                "Install a CUDA-enabled torch wheel to train on GPU."
            )
    except Exception as exc:
        logger.warning(f"CUDA detection failed ({exc}); falling back to CPU")

    logger.info("Using CPU")
    return "cpu"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base YOLO variant (yolo11n.pt, yolo11s.pt, …)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data-dir", default="data/annotated")
    parser.add_argument("--device", type=str, default=None,
                        help="Training device: e.g. '0', 'cpu', '0,1'. Auto-detect if omitted.")
    args = parser.parse_args()

    train(
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        resume=args.resume,
        annotated_dir=args.data_dir,
        device=args.device,
    )
