"""
scripts/collect_data.py
========================
Automatically collect training screenshots from sudoku.com.

For each puzzle:
  1. Open sudoku.com in a headless browser
  2. Accept cookies
  3. Take a screenshot
  4. Save to data/raw/

You can also collect solved grids (after the puzzle is complete) to get
examples of cells containing digits — useful for balanced training data.

Usage:
    python scripts/collect_data.py --count 200 --difficulty easy
    python scripts/collect_data.py --count 200 --difficulty hard
"""

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DIFFICULTIES = ["easy", "medium", "hard", "expert"]
SUDOKU_URLS = {
    "easy":   "https://sudoku.com/easy/",
    "medium": "https://sudoku.com/medium/",
    "hard":   "https://sudoku.com/hard/",
    "expert": "https://sudoku.com/expert/",
}


def collect(count: int = 100, difficulty: str = "easy", output_dir: str = "data/raw"):
    from playwright.sync_api import sync_playwright
    output = Path(output_dir) / difficulty
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting {count} screenshots — difficulty: {difficulty}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True,
                                     args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        for i in range(count):
            try:
                page.goto(SUDOKU_URLS[difficulty], wait_until="networkidle", timeout=30_000)
                page.wait_for_timeout(1500)

                # Accept cookies on first page load
                if i == 0:
                    _accept_cookies(page)

                # Screenshot the full page
                path = output / f"{difficulty}_{i:04d}.png"
                page.screenshot(path=str(path))
                logger.info(f"[{i+1}/{count}] Saved {path.name}")

                # Navigate to a new puzzle (click "New game" if present)
                try:
                    page.locator("button:has-text('New Game')").first.click(timeout=2000)
                    page.wait_for_timeout(800)
                except Exception:
                    pass

            except Exception as e:
                logger.warning(f"Failed on screenshot {i}: {e}")
                time.sleep(2)

        browser.close()
    logger.info(f"Done — {count} screenshots saved to {output}")


def _accept_cookies(page):
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button.fc-button.fc-cta-consent",
        "button:has-text('Accept')",
    ]
    for sel in selectors:
        try:
            page.locator(sel).first.click(timeout=2000)
            return
        except Exception:
            continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default="easy")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    collect(args.count, args.difficulty, args.output_dir)
