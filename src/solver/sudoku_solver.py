"""
solver/sudoku_solver.py
========================
Sudoku solving engine with two strategies:
  1. Constraint propagation (arc consistency / naked singles / hidden singles)
  2. Backtracking with MRV heuristic (Minimum Remaining Values)

Typical solve time: < 1 ms for easy/medium, < 5 ms for hard/expert puzzles.
"""

import copy
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SudokuSolver:
    """
    Solves a 9×9 Sudoku grid.

    Parameters
    ----------
    grid : list[list[int]]
        9×9 matrix. 0 = empty cell, 1–9 = given digit.

    Returns solved 9×9 matrix, or None if no solution exists.
    """

    def solve(self, grid: list[list[int]]) -> Optional[list[list[int]]]:
        """
        Main entry point. Tries constraint propagation first,
        then falls back to backtracking if needed.
        """
        board = copy.deepcopy(grid)
        t0 = time.perf_counter()

        # Validate grid before solving
        validation_errors = self._validate_grid(board)
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Grid validation: {error}")
            logger.error("Grid is invalid — digit recognition may have errors.")
            return None

        # Phase 1: constraint propagation
        if not self._propagate(board):
            logger.error("Grid is unsolvable (constraint propagation detected contradiction)")
            return None

        # Phase 2: backtracking (for harder puzzles)
        result = self._backtrack(board)
        elapsed = (time.perf_counter() - t0) * 1000
        if result:
            logger.info(f"Solved in {elapsed:.3f} ms")
        else:
            logger.error("No solution found")
        return result

    def _validate_grid(self, grid: list[list[int]]) -> list[str]:
        """
        Validate the grid for common errors from digit recognition.
        Returns list of error messages (empty if valid).
        """
        errors = []

        # Check dimensions
        if len(grid) != 9 or not all(len(row) == 9 for row in grid):
            errors.append("Grid must be 9×9")
            return errors

        # Check value ranges
        for r in range(9):
            for c in range(9):
                val = grid[r][c]
                if not isinstance(val, int) or val < 0 or val > 9:
                    errors.append(f"Invalid value at ({r},{c}): {val}")

        # Check for duplicate values in rows
        for r in range(9):
            non_zero = [v for v in grid[r] if v != 0]
            if len(non_zero) != len(set(non_zero)):
                duplicates = [v for v in set(non_zero) if non_zero.count(v) > 1]
                errors.append(f"Row {r} has duplicate values: {duplicates}")

        # Check for duplicate values in columns
        for c in range(9):
            non_zero = [grid[r][c] for r in range(9) if grid[r][c] != 0]
            if len(non_zero) != len(set(non_zero)):
                duplicates = [v for v in set(non_zero) if non_zero.count(v) > 1]
                errors.append(f"Column {c} has duplicate values: {duplicates}")

        # Check for duplicate values in 3×3 boxes
        for box_r in range(3):
            for box_c in range(3):
                non_zero = []
                for r in range(3):
                    for c in range(3):
                        val = grid[box_r * 3 + r][box_c * 3 + c]
                        if val != 0:
                            non_zero.append(val)
                if len(non_zero) != len(set(non_zero)):
                    duplicates = [v for v in set(non_zero) if non_zero.count(v) > 1]
                    errors.append(f"Box ({box_r},{box_c}) has duplicate values: {duplicates}")

        return errors

    # ──────────────────────────────────────────────────────────────────────
    # Constraint propagation
    # ──────────────────────────────────────────────────────────────────────

    def _propagate(self, board: list[list[int]]) -> bool:
        """
        Repeatedly apply naked singles and hidden singles until stable.
        Returns False if a contradiction is found.
        """
        changed = True
        while changed:
            changed = False
            candidates = self._compute_candidates(board)

            for r in range(9):
                for c in range(9):
                    if board[r][c] != 0:
                        continue
                    cands = candidates[r][c]
                    if len(cands) == 0:
                        return False  # contradiction
                    if len(cands) == 1:
                        board[r][c] = next(iter(cands))
                        changed = True

            # Hidden singles in rows, columns, boxes
            if self._apply_hidden_singles(board, candidates):
                changed = True

        return True

    def _compute_candidates(self, board: list[list[int]]) -> list[list[set]]:
        candidates = [[set(range(1, 10)) if board[r][c] == 0 else set()
                       for c in range(9)] for r in range(9)]
        for r in range(9):
            for c in range(9):
                if board[r][c] != 0:
                    self._eliminate(candidates, r, c, board[r][c])
        return candidates

    @staticmethod
    def _eliminate(candidates: list[list[set]], row: int, col: int, val: int):
        # Row
        for c in range(9):
            candidates[row][c].discard(val)
        # Column
        for r in range(9):
            candidates[r][col].discard(val)
        # 3×3 box
        br, bc = (row // 3) * 3, (col // 3) * 3
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                candidates[r][c].discard(val)

    @staticmethod
    def _apply_hidden_singles(
        board: list[list[int]], candidates: list[list[set]]
    ) -> bool:
        """
        A hidden single is a digit that can only go in one cell
        within a row, column, or box.
        """
        changed = False
        # Check rows, cols, boxes
        units = []
        for i in range(9):
            units.append([(i, c) for c in range(9)])       # row
            units.append([(r, i) for r in range(9)])       # col
            br, bc = (i // 3) * 3, (i % 3) * 3
            units.append([(br + dr, bc + dc)
                          for dr in range(3) for dc in range(3)])  # box

        for unit in units:
            for digit in range(1, 10):
                possible = [(r, c) for r, c in unit
                            if digit in candidates[r][c]]
                if len(possible) == 1:
                    r, c = possible[0]
                    if board[r][c] == 0:
                        board[r][c] = digit
                        candidates[r][c] = set()
                        SudokuSolver._eliminate(candidates, r, c, digit)
                        changed = True
        return changed

    # ──────────────────────────────────────────────────────────────────────
    # Backtracking with MRV heuristic
    # ──────────────────────────────────────────────────────────────────────

    def _backtrack(self, board: list[list[int]]) -> Optional[list[list[int]]]:
        """
        Recursive backtracking. Uses MRV (minimum remaining values):
        always fills the cell with the fewest candidates first.
        """
        empty = self._find_empty_mrv(board)
        if empty is None:
            return board  # solved!

        r, c = empty
        candidates = self._get_candidates(board, r, c)
        for val in sorted(candidates):
            board[r][c] = val
            result = self._backtrack(board)
            if result is not None:
                return result
            board[r][c] = 0  # undo

        return None  # trigger backtrack

    @staticmethod
    def _find_empty_mrv(board: list[list[int]]) -> Optional[tuple[int, int]]:
        best_cell = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if board[r][c] != 0:
                    continue
                used = set()
                for col in range(9):
                    used.add(board[r][col])
                for row in range(9):
                    used.add(board[row][c])
                br, bc = (r // 3) * 3, (c // 3) * 3
                for dr in range(3):
                    for dc in range(3):
                        used.add(board[br + dr][bc + dc])
                count = 9 - len(used) + 1  # +1 for the 0 itself
                if count < best_count:
                    best_count = count
                    best_cell = (r, c)
        return best_cell

    @staticmethod
    def _get_candidates(board: list[list[int]], r: int, c: int) -> set[int]:
        used: set[int] = set()
        used.update(board[r])                          # row
        used.update(board[row][c] for row in range(9)) # col
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                used.add(board[br + dr][bc + dc])
        return set(range(1, 10)) - used

    # ──────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def is_valid(grid: list[list[int]]) -> bool:
        """Check that a completed grid satisfies all Sudoku constraints."""
        for i in range(9):
            row = [v for v in grid[i] if v != 0]
            col = [grid[r][i] for r in range(9) if grid[r][i] != 0]
            if len(row) != len(set(row)) or len(col) != len(set(col)):
                return False
            br, bc = (i // 3) * 3, (i % 3) * 3
            box = [grid[br + dr][bc + dc]
                   for dr in range(3) for dc in range(3)
                   if grid[br + dr][bc + dc] != 0]
            if len(box) != len(set(box)):
                return False
        return True
