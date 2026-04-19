"""
scripts/extract_templates.py
==============================
Extract digit templates from a solved sudoku.com screenshot.

Given a screenshot where the grid solution is visible,
and the known solution matrix, crop each digit cell and save
to data/templates/{digit}_{row}_{col}.png.

Usage:
    python scripts/extract_templates.py \
        --screenshot data/raw/solved_example.png \
        --solution "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_templates(
    screenshot_path: str,
    solution_str: str,
    output_dir: str = "data/templates",
    grid_box: tuple = None,
):
    """
    Parameters
    ----------
    screenshot_path : str
        Path to screenshot with visible solved grid.
    solution_str : str
        81-character string of the solution, row by row.
        Use '0' for empty cells (shouldn't be any in a solved grid).
    output_dir : str
        Where to save the cropped templates.
    grid_box : tuple (x1,y1,x2,y2), optional
        If provided, use this as the grid bounding box.
        Otherwise, auto-detect using contour analysis.
    """
    assert len(solution_str) == 81, "Solution must be 81 characters"
    solution = [int(c) for c in solution_str]
    grid = [solution[i*9:(i+1)*9] for i in range(9)]

    img = cv2.imread(screenshot_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {screenshot_path}")

    if grid_box is None:
        grid_box = _auto_detect_grid(img)
    x1, y1, x2, y2 = grid_box
    cell_w = (x2 - x1) / 9
    cell_h = (y2 - y1) / 9

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0

    for r in range(9):
        for c in range(9):
            digit = grid[r][c]
            if digit == 0:
                continue
            cx1 = int(x1 + c * cell_w)
            cy1 = int(y1 + r * cell_h)
            cx2 = int(cx1 + cell_w)
            cy2 = int(cy1 + cell_h)

            crop = img[cy1:cy2, cx1:cx2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            pad = max(2, gray.shape[0] // 10)
            gray = gray[pad:-pad, pad:-pad]
            resized = cv2.resize(gray, (40, 40))

            path = out / f"{digit}_{r}{c}.png"
            cv2.imwrite(str(path), resized)
            saved += 1

    logger.info(f"Saved {saved} templates to {out}")


def _auto_detect_grid(img: np.ndarray) -> tuple:
    """Simple contour-based grid detection for template extraction."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find largest near-square contour
    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / h if h > 0 else 0
        if area > best_area and 0.85 < aspect < 1.15 and w > 200:
            best_area = area
            best = (x, y, x + w, y + h)
    if best is None:
        h, w = img.shape[:2]
        return (0, 0, w, h)
    return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--output-dir", default="data/templates")
    args = parser.parse_args()
    extract_templates(args.screenshot, args.solution, args.output_dir)
