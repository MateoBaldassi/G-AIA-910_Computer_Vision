"""
Sudoku Solver — Computer Vision Pipeline
=========================================
Entry point. Runs the full pipeline:
  1. Screenshot the browser
  2. Detect & localize the grid + cells (YOLO)
  3. Recognize digits (configurable approach)
  4. Solve the puzzle (backtracking)
  5. Interact with the browser to fill in the solution

Usage:
    python src/main.py --digit-method easyocr
    python src/main.py --digit-method tesseract
    python src/main.py --digit-method template
    python src/main.py --digit-method cnn
    python src/main.py --benchmark          # compare all digit methods
"""

import argparse
import time
import logging

from detection.grid_detector import GridDetector
from ocr.digit_recognizer import DigitRecognizer
from solver.sudoku_solver import SudokuSolver
from automation.browser_controller import BrowserController
from utils.logger import setup_logger
from utils.metrics import PipelineMetrics

logger = logging.getLogger(__name__)


def run_pipeline(digit_method: str = "easyocr", headless: bool = False) -> dict:
    """
    Run the full Sudoku solving pipeline.

    Returns a dict with timing/accuracy metrics for benchmarking.
    """
    metrics = PipelineMetrics()

    # --- Step 1 : Launch browser and navigate to sudoku.com ---
    logger.info("=== Step 1: Browser setup ===")
    controller = BrowserController(headless=headless)
    controller.open_sudoku_site()
    controller.accept_cookies()          # HTML interaction — allowed by spec

    # --- Step 2 : Screenshot + grid detection ---
    logger.info("=== Step 2: Visual grid detection ===")
    time.sleep(5.0)
    t0 = time.perf_counter()
    screenshot = controller.take_screenshot()

    detector = GridDetector(model_path="models/yolo_sudoku.pt")
    grid_coords, cell_images = detector.detect(screenshot)
    metrics.detection_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Detection: {metrics.detection_ms:.1f} ms — {len(cell_images)} cells found")

    # --- Step 3 : Digit recognition ---
    logger.info(f"=== Step 3: Digit recognition ({digit_method}) ===")
    t0 = time.perf_counter()
    recognizer = DigitRecognizer(method=digit_method)
    grid_state = recognizer.recognize_grid(cell_images)   # 9×9 list, 0 = empty
    metrics.recognition_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Recognition: {metrics.recognition_ms:.1f} ms")
    logger.info(f"Grid parsed:\n{_format_grid(grid_state)}")

    # --- Step 4 : Solve ---
    logger.info("=== Step 4: Solving ===")
    t0 = time.perf_counter()
    solver = SudokuSolver()
    solution = solver.solve(grid_state)
    metrics.solving_ms = (time.perf_counter() - t0) * 1000

    if solution is None:
        logger.error("No solution found — digit recognition may have errors.")
        controller.close()
        return metrics.to_dict()

    logger.info(f"Solved in {metrics.solving_ms:.1f} ms")
    logger.info(f"Solution:\n{_format_grid(solution)}")

    # --- Step 5 : Interact ---
    logger.info("=== Step 5: Filling in solution ===")
    t0 = time.perf_counter()
    controller.fill_solution(grid_coords, grid_state, solution)
    metrics.interaction_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Interaction: {metrics.interaction_ms:.1f} ms")

    controller.close()
    metrics.success = True
    logger.info(f"=== Pipeline complete — total {metrics.total_ms:.1f} ms ===")
    return metrics.to_dict()


def run_benchmark():
    """Compare all digit recognition methods under identical conditions."""
    from utils.metrics import BenchmarkReport
    methods = ["easyocr", "tesseract", "template", "cnn"]
    results = {}
    for method in methods:
        logger.info(f"\n{'='*50}\nBenchmarking: {method}\n{'='*50}")
        try:
            m = run_pipeline(digit_method=method, headless=True)
            results[method] = m
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            results[method] = {"success": False, "error": str(e)}

    report = BenchmarkReport(results)
    report.print_table()
    report.save_csv("benchmark_results.csv")


def _format_grid(grid: list[list[int]]) -> str:
    lines = []
    for i, row in enumerate(grid):
        if i % 3 == 0 and i > 0:
            lines.append("------+-------+------")
        cells = []
        for j, val in enumerate(row):
            if j % 3 == 0 and j > 0:
                cells.append("|")
            cells.append(str(val) if val != 0 else ".")
        lines.append(" ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="CV Sudoku Solver")
    parser.add_argument(
        "--digit-method",
        choices=["easyocr", "tesseract", "template", "cnn"],
        default="easyocr",
        help="Digit recognition approach",
    )
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all methods")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    else:
        run_pipeline(digit_method=args.digit_method, headless=args.headless)
