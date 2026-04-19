"""
detection/grid_detector.py
===========================
YOLO-based detection of the Sudoku grid and individual cells.

The model is fine-tuned on screenshots of sudoku.com.
It detects two classes:
    - class 0 : "cell"   — individual cells (81 detections per image)
    - class 1 : "grid"   — the full 9×9 bounding box

All perception is purely visual (no DOM/HTML access).
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class GridDetector:
    """
    Detects the Sudoku grid and extracts individual cell images using YOLO.

    Parameters
    ----------
    model_path : str
        Path to the fine-tuned YOLO weights file (.pt).
    conf_threshold : float
        Minimum confidence for detections.
    """

    CELL_CLASS = 0
    GRID_CLASS = 1

    def __init__(self, model_path: str = "models/yolo_sudoku.pt", conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"YOLO model not found at '{model_path}'.\n"
                "Train the model first: python scripts/train_yolo.py"
            )
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self._configure_class_ids_from_model()

    def _configure_class_ids_from_model(self):
        """Map class IDs from YOLO metadata so either label order works."""
        names = getattr(self.model, "names", None)
        if not isinstance(names, dict):
            logger.warning("YOLO class names unavailable, using default IDs cell=0 grid=1")
            return

        # Normalize mapping like {0: 'cell', 1: 'grid'}.
        normalized = {int(k): str(v).strip().lower() for k, v in names.items()}
        inv = {v: k for k, v in normalized.items()}

        if "cell" in inv and "grid" in inv:
            self.CELL_CLASS = inv["cell"]
            self.GRID_CLASS = inv["grid"]
            logger.info(
                "YOLO class mapping detected from model: cell=%d grid=%d",
                self.CELL_CLASS,
                self.GRID_CLASS,
            )
        else:
            logger.warning(
                "YOLO class names do not include both 'cell' and 'grid' (%s). "
                "Using defaults cell=0 grid=1.",
                normalized,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, screenshot: np.ndarray) -> tuple[list[list[dict]], list[list[np.ndarray]]]:
        """
        Run detection on a screenshot.

        Parameters
        ----------
        screenshot : np.ndarray
            BGR image from the browser screenshot.

        Returns
        -------
        grid_coords : list[list[dict]]
            9×9 grid of cell bounding boxes: {"x1", "y1", "x2", "y2", "cx", "cy"}
        cell_images : list[list[np.ndarray]]
            9×9 grid of cropped cell images (grayscale, 64×64).
        """
        results = self.model(screenshot, conf=self.conf_threshold, verbose=False)[0]
        boxes = results.boxes

        grid_box = self._extract_grid_box(boxes, screenshot)
        cell_boxes = self._extract_cell_boxes(boxes, grid_box)
        grid_coords = self._sort_cells_into_grid(cell_boxes, grid_box)
        cell_images = self._crop_cells(screenshot, grid_coords)

        return grid_coords, cell_images

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_grid_box(self, boxes, image: np.ndarray) -> dict:
        """Find the full-grid bounding box using GRID_CLASS."""
        grid_detections = [
            b for b in boxes
            if int(b.cls) == self.GRID_CLASS and float(b.conf) >= self.conf_threshold
        ]
        if not grid_detections:
            logger.warning("No grid detected — falling back to full image")
            h, w = image.shape[:2]
            return {"x1": 0, "y1": 0, "x2": w, "y2": h}

        # Take the highest-confidence grid box
        best = max(grid_detections, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _extract_cell_boxes(self, boxes, grid_box: dict) -> list[dict]:
        """Extract all cell detections using CELL_CLASS within the grid area."""
        cells = []
        for b in boxes:
            if int(b.cls) != self.CELL_CLASS:
                continue
            if float(b.conf) < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Filter to cells inside the grid bounding box
            if (grid_box["x1"] <= cx <= grid_box["x2"] and
                    grid_box["y1"] <= cy <= grid_box["y2"]):
                cells.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "cx": cx, "cy": cy})
        return cells

    def _sort_cells_into_grid(self, cells: list[dict], grid_box: dict) -> list[list[dict]]:
        """
        Sort 81 detected cells into a 9×9 grid.

        Strategy: cluster by y-coordinate (rows), then sort each row by x.
        Falls back to uniform grid subdivision if fewer than 81 cells detected.
        """
        if len(cells) < 81:
            logger.warning(
                f"Only {len(cells)} cells detected (expected 81). "
                "Switching to uniform subdivision fallback."
            )
            return self._fallback_uniform_grid(cells, grid_box)

        # Sort all cells by y, then group into 9 rows of 9
        cells_sorted_y = sorted(cells, key=lambda c: c["cy"])
        rows = []
        for i in range(9):
            row_cells = cells_sorted_y[i * 9: (i + 1) * 9]
            row_sorted_x = sorted(row_cells, key=lambda c: c["cx"])
            rows.append(row_sorted_x)
        return rows

    def _fallback_uniform_grid(self, cells: list[dict], grid_box: dict) -> list[list[dict]]:
        """
        If fewer than 81 cells are detected, infer a uniform 9×9 grid
        from the bounding boxes of the detected cells.
        """
        if cells:
            x1s = [c["x1"] for c in cells]
            y1s = [c["y1"] for c in cells]
            x2s = [c["x2"] for c in cells]
            y2s = [c["y2"] for c in cells]
            gx1, gy1 = min(x1s), min(y1s)
            gx2, gy2 = max(x2s), max(y2s)
        else:
            # Final fallback: use detected grid box (or full image fallback grid box).
            gx1, gy1 = grid_box["x1"], grid_box["y1"]
            gx2, gy2 = grid_box["x2"], grid_box["y2"]
        cell_w = (gx2 - gx1) / 9
        cell_h = (gy2 - gy1) / 9

        grid = []
        for row in range(9):
            row_cells = []
            for col in range(9):
                x1 = int(gx1 + col * cell_w)
                y1 = int(gy1 + row * cell_h)
                x2 = int(x1 + cell_w)
                y2 = int(y1 + cell_h)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                row_cells.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                   "cx": cx, "cy": cy})
            grid.append(row_cells)
        return grid

    def _crop_cells(
        self, image: np.ndarray, grid_coords: list[list[dict]]
    ) -> list[list[np.ndarray]]:
        """
        Crop each cell from the screenshot and normalize to 64×64 grayscale.
        Applies light preprocessing (threshold, denoise) to improve OCR accuracy.
        """
        cell_images = []
        for row in grid_coords:
            row_imgs = []
            for cell in row:
                crop = image[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]]
                if crop.size == 0:
                    row_imgs.append(np.zeros((64, 64), dtype=np.uint8))
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Slight padding removal (border artifacts from grid lines)
                pad = max(2, gray.shape[0] // 10)
                gray = gray[pad:-pad, pad:-pad] if gray.shape[0] > 2 * pad else gray
                # Adaptive threshold to binarize
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                resized = cv2.resize(binary, (64, 64), interpolation=cv2.INTER_AREA)
                row_imgs.append(resized)
            cell_images.append(row_imgs)
        return cell_images
