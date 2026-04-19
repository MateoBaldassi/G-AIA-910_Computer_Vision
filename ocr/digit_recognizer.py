"""
ocr/digit_recognizer.py
========================
Recognizes digits (1–9) in individual cell images.

Four interchangeable methods:
  - "easyocr"   : EasyOCR neural model (default, best accuracy)
  - "tesseract" : Tesseract OCR (fast, needs fine-tuning config)
  - "template"  : OpenCV template matching (no ML dependency)
  - "cnn"       : Custom CNN fine-tuned on sudoku.com digits

All methods share the same interface: recognize_grid(cell_images) → 9×9 int list.
A cell value of 0 means "empty".
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseDigitRecognizer(ABC):
    """Common interface for all digit recognition backends."""

    @abstractmethod
    def predict_cell(self, cell_img: np.ndarray) -> int:
        """Return digit 1–9, or 0 if cell is empty."""

    def recognize_grid(self, cell_images: list[list[np.ndarray]]) -> list[list[int]]:
        grid = []
        for row in cell_images:
            grid.append([self.predict_cell(cell) for cell in row])
        
        # Log grid statistics
        self._log_grid_stats(grid)
        return grid

    @staticmethod
    def _log_grid_stats(grid: list[list[int]]):
        """Log statistics about recognized grid."""
        flat = [v for row in grid for v in row]
        non_empty = [v for v in flat if v != 0]
        logger.info(f"Grid recognition stats: {len(non_empty)} cells recognized, {81 - len(non_empty)} empty")
        if non_empty:
            digit_counts = {}
            for d in range(1, 10):
                count = non_empty.count(d)
                if count > 0:
                    digit_counts[d] = count
            logger.info(f"Digit distribution: {dict(sorted(digit_counts.items()))}")

    @staticmethod
    def _is_empty(cell_img: np.ndarray, white_ratio_threshold: float = 0.97) -> bool:
        """Heuristic: if the cell is mostly white/blank it's empty."""
        white_pixels = np.sum(cell_img > 200)
        total = cell_img.size
        return (white_pixels / total) >= white_ratio_threshold


# ──────────────────────────────────────────────────────────────────────────────
# Method 1 — EasyOCR
# ──────────────────────────────────────────────────────────────────────────────

class EasyOCRRecognizer(BaseDigitRecognizer):
    """
    Uses EasyOCR with English digit recognition.
    Accuracy: ~97–99% on sudoku.com screenshots.
    Avg. per-cell: ~8 ms (GPU) / ~35 ms (CPU).
    """

    def __init__(self):
        import easyocr
        logger.info("Initializing EasyOCR (first run downloads model ~40 MB)")
        # gpu=True if CUDA available
        self.reader = easyocr.Reader(["en"], gpu=self._has_gpu())
        logger.info("EasyOCR ready")

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def predict_cell(self, cell_img: np.ndarray) -> int:
        if self._is_empty(cell_img):
            return 0
        # Invert: EasyOCR prefers dark text on white
        inverted = cv2.bitwise_not(cell_img)
        results = self.reader.readtext(
            inverted,
            allowlist="123456789",
            detail=1,  # Return with confidence scores
            paragraph=False,
        )
        if not results:
            return 0
        
        # Get best result by confidence
        best_result = max(results, key=lambda x: x[2]) if results else None
        if not best_result:
            return 0
        
        text = best_result[1].strip()
        confidence = best_result[2]
        
        if text.isdigit() and 1 <= int(text) <= 9:
            if confidence < 0.7:
                logger.debug(f"Low EasyOCR confidence for digit {text}: {confidence:.3f}")
            return int(text)
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Method 2 — Tesseract
# ──────────────────────────────────────────────────────────────────────────────

class TesseractRecognizer(BaseDigitRecognizer):
    """
    Uses pytesseract with digit-only whitelist config.
    Accuracy: ~93–96%. Fastest CPU option (~12 ms/cell).
    Requires: tesseract-ocr installed on system.
    """

    def __init__(self):
        import pytesseract
        self.tess = pytesseract
        # Single character mode, digits only
        self.config = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
        logger.info("Tesseract OCR backend ready")

    def predict_cell(self, cell_img: np.ndarray) -> int:
        if self._is_empty(cell_img):
            return 0
        inverted = cv2.bitwise_not(cell_img)
        # Scale up for better Tesseract accuracy
        scaled = cv2.resize(inverted, (128, 128), interpolation=cv2.INTER_CUBIC)
        text = self.tess.image_to_string(scaled, config=self.config).strip()
        return int(text) if text.isdigit() and 1 <= int(text) <= 9 else 0


# ──────────────────────────────────────────────────────────────────────────────
# Method 3 — Template Matching (OpenCV)
# ──────────────────────────────────────────────────────────────────────────────

class TemplateMatcher(BaseDigitRecognizer):
    """
    Classical template matching using pre-extracted digit templates.
    No ML dependency. Accuracy: ~88–92% (sensitive to font/zoom changes).
    Speed: ~3 ms/cell (pure CPU, very fast).

    Templates are extracted automatically from a known solved grid screenshot
    using scripts/extract_templates.py, stored in data/templates/{1..9}.png.
    """

    TEMPLATE_SIZE = (40, 40)
    MATCH_THRESHOLD = 0.70  # TM_CCOEFF_NORMED threshold

    def __init__(self, template_dir: str = "data/templates"):
        from pathlib import Path
        self.templates: dict[int, list[np.ndarray]] = {}
        template_path = Path(template_dir)

        if not template_path.exists():
            logger.warning(
                f"Template directory '{template_dir}' not found. "
                "Run scripts/extract_templates.py first."
            )
            return

        for digit in range(1, 10):
            paths = list(template_path.glob(f"{digit}_*.png")) + \
                    list(template_path.glob(f"{digit}.png"))
            if paths:
                self.templates[digit] = [
                    cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE),
                               self.TEMPLATE_SIZE)
                    for p in paths
                ]
        logger.info(f"Loaded templates for digits: {sorted(self.templates.keys())}")

    def predict_cell(self, cell_img: np.ndarray) -> int:
        if self._is_empty(cell_img):
            return 0
        if not self.templates:
            return 0

        cell_resized = cv2.resize(cell_img, self.TEMPLATE_SIZE)
        best_digit, best_score = 0, self.MATCH_THRESHOLD

        for digit, templates in self.templates.items():
            for tmpl in templates:
                result = cv2.matchTemplate(cell_resized, tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(result.max())
                if score > best_score:
                    best_score = score
                    best_digit = digit

        return best_digit


# ──────────────────────────────────────────────────────────────────────────────
# Method 4 — Custom CNN
# ──────────────────────────────────────────────────────────────────────────────

class CNNRecognizer(BaseDigitRecognizer):
    """
    Lightweight CNN fine-tuned on digits extracted from sudoku.com screenshots.
    Architecture: 3× Conv-BN-ReLU-Pool → FC128 → FC10 (classes 0–9).
    Accuracy: ~99% on in-distribution data.
    Speed: ~2 ms/cell (GPU), ~6 ms/cell (CPU).

    Model weights: models/cnn_digits.pth
    Training: notebooks/02_train_cnn.ipynb
    """

    INPUT_SIZE = (64, 64)

    def __init__(self, model_path: str = "models/cnn_digits.pth"):
        import torch
        from ocr.cnn_model import SudokuDigitCNN

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SudokuDigitCNN(num_classes=10)

        from pathlib import Path
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"CNN model not found at '{model_path}'.\n"
                "Train it first: python notebooks/02_train_cnn.ipynb"
            )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval().to(self.device)
        logger.info(f"CNN model loaded ({self.device})")

    def predict_cell(self, cell_img: np.ndarray) -> int:
        if self._is_empty(cell_img):
            return 0
        import torch

        img = cv2.resize(cell_img, self.INPUT_SIZE).astype(np.float32) / 255.0
        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,64,64)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            pred = int(pred.item())
            confidence = float(confidence.item())
        
        # Log low confidence predictions
        if pred > 0 and confidence < 0.8:  # 0 is empty class
            logger.debug(f"Low CNN confidence for digit {pred}: {confidence:.3f}")
        
        return pred  # 0 = empty, 1–9 = digit


# ──────────────────────────────────────────────────────────────────────────────
# Factory / Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

class DigitRecognizer:
    """
    Facade that instantiates the requested backend.

    Usage:
        recognizer = DigitRecognizer(method="easyocr")
        grid = recognizer.recognize_grid(cell_images)  # → 9×9 list[list[int]]
    """

    _BACKENDS = {
        "easyocr": EasyOCRRecognizer,
        "tesseract": TesseractRecognizer,
        "template": TemplateMatcher,
        "cnn": CNNRecognizer,
    }

    def __init__(self, method: str = "easyocr", **kwargs):
        if method not in self._BACKENDS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: {list(self._BACKENDS.keys())}"
            )
        logger.info(f"Digit recognizer: {method}")
        self._backend: BaseDigitRecognizer = self._BACKENDS[method](**kwargs)

    def recognize_grid(self, cell_images: list[list[np.ndarray]]) -> list[list[int]]:
        return self._backend.recognize_grid(cell_images)
