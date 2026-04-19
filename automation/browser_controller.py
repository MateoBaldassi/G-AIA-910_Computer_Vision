"""
automation/browser_controller.py
==================================
Playwright-based browser automation for sudoku.com.

IMPORTANT: Per project spec, HTML/DOM access is used ONLY for:
  - Cookie banner acceptance
  - Clicking the "OK" / validation button

All grid perception is done via screenshots (computer vision).
Digit entry uses keyboard simulation after clicking the detected cell coordinates.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Delay between keystrokes to avoid bot detection
KEYSTROKE_DELAY_MS = 80
CLICK_DELAY_MS = 50


class BrowserController:
    """
    Controls a Chromium browser to:
      1. Navigate to sudoku.com
      2. Accept cookie banners (HTML — permitted by spec)
      3. Take screenshots
      4. Fill in solved digits by clicking cell coordinates + keypress

    Parameters
    ----------
    headless : bool
        Run browser without GUI (for Docker/CI environments).
    slow_mo : int
        Extra delay (ms) between Playwright actions.
    """

    SUDOKU_URL = "https://sudoku.com/"

    def __init__(self, headless: bool = False, slow_mo: int = 0):
        from playwright.sync_api import sync_playwright
        self._pw_context = sync_playwright()
        self._pw = self._pw_context.__enter__()
        self._browser = self._pw.chromium.launch(
            headless=headless,
            slow_mo=slow_mo,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self._page = self._browser.new_page(
            viewport={"width": 1280, "height": 900}
        )
        logger.info("Browser launched")

    # ──────────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────────

    def open_sudoku_site(self):
        logger.info(f"Navigating to {self.SUDOKU_URL}")
        self._page.goto(self.SUDOKU_URL, wait_until="networkidle", timeout=30_000)
        self._page.wait_for_timeout(1500)

    def accept_cookies(self):
        """
        Accept cookie banners via HTML interaction.
        Permitted by spec: 'ancillary interaction unrelated to grid perception'.
        """
        selectors = [
            "button#onetrust-accept-btn-handler",
            "button.fc-button.fc-cta-consent",
            "button[aria-label*='Accept']",
            "button:has-text('Accept')",
            "button:has-text('Agree')",
            "button:has-text('OK')",
        ]
        for sel in selectors:
            try:
                self._page.locator(sel).first.click(timeout=2000)
                logger.info(f"Cookie banner dismissed ({sel})")
                self._page.wait_for_timeout(500)
                return
            except Exception:
                continue
        logger.info("No cookie banner found (or already dismissed)")

    # ──────────────────────────────────────────────────────────────────────
    # Screenshot
    # ──────────────────────────────────────────────────────────────────────

    def take_screenshot(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        Capture the current page as a numpy BGR array.

        Parameters
        ----------
        save_path : str, optional
            If provided, also save the screenshot to disk.
        """
        import cv2
        png_bytes = self._page.screenshot(full_page=False)
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if save_path:
            cv2.imwrite(save_path, img)
            logger.debug(f"Screenshot saved: {save_path}")

        return img

    # ──────────────────────────────────────────────────────────────────────
    # Solution entry
    # ──────────────────────────────────────────────────────────────────────

    def fill_solution(
        self,
        grid_coords: list[list[dict]],
        original_grid: list[list[int]],
        solution: list[list[int]],
    ):
        """
        Click each empty cell (by pixel coordinates) and type the solution digit.

        Parameters
        ----------
        grid_coords : list[list[dict]]
            9×9 grid of cell bounding boxes with "cx"/"cy" center coordinates.
        original_grid : list[list[int]]
            The parsed grid (0 = empty cells that need filling).
        solution : list[list[int]]
            The fully solved 9×9 grid.
        """
        cells_filled = 0
        for r in range(9):
            for c in range(9):
                if original_grid[r][c] != 0:
                    continue  # Pre-filled by puzzle — skip
                digit = solution[r][c]
                cell = grid_coords[r][c]
                cx, cy = cell["cx"], cell["cy"]

                self._page.mouse.click(cx, cy)
                time.sleep(CLICK_DELAY_MS / 1000)
                self._page.keyboard.press(str(digit))
                time.sleep(KEYSTROKE_DELAY_MS / 1000)
                cells_filled += 1

        logger.info(f"Filled {cells_filled} cells")

    # ──────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────

    def close(self):
        try:
            self._browser.close()
            self._pw_context.__exit__(None, None, None)
            logger.info("Browser closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
