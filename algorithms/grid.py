"""Grid-world (maze) environment with four-directional movement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int]  # (row, col)

# Fixed neighbour order (up, right, down, left) so that tie-breaking is deterministic.
MOVES: Tuple[Cell, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class Grid:
    rows: int
    cols: int
    start: Cell
    goal: Cell
    obstacles: Set[Cell] = field(default_factory=set)

    # ---- geometry -----------------------------------------------------
    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.rows and 0 <= c[1] < self.cols

    def passable(self, c: Cell) -> bool:
        return self.in_bounds(c) and c not in self.obstacles

    def neighbors(self, c: Cell) -> List[Cell]:
        """Passable 4-neighbours in the fixed order up, right, down, left."""
        out = []
        for dr, dc in MOVES:
            n = (c[0] + dr, c[1] + dc)
            if self.passable(n):
                out.append(n)
        return out

    def manhattan(self, a: Cell, b: Cell) -> int:
        """Manhattan distance: the consistent heuristic for 4-directional unit-cost moves."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---- validation ---------------------------------------------------
    def validate(self) -> List[str]:
        """Return human-readable problems (empty list = valid)."""
        errs: List[str] = []
        if self.rows < 2 or self.cols < 2:
            errs.append("The grid must be at least 2 x 2.")
        if not self.in_bounds(self.start):
            errs.append("The start cell is outside the grid.")
        elif self.start in self.obstacles:
            errs.append("The start cell is an obstacle.")
        if not self.in_bounds(self.goal):
            errs.append("The goal cell is outside the grid.")
        elif self.goal in self.obstacles:
            errs.append("The goal cell is an obstacle.")
        return errs

    # ---- editing helpers (used by the UI) -----------------------------
    def set_start(self, c: Cell) -> bool:
        if not self.in_bounds(c) or c == self.goal:
            return False
        self.obstacles.discard(c)
        self.start = c
        return True

    def set_goal(self, c: Cell) -> bool:
        if not self.in_bounds(c) or c == self.start:
            return False
        self.obstacles.discard(c)
        self.goal = c
        return True

    def toggle_obstacle(self, c: Cell) -> bool:
        """Add/remove an obstacle. The start and goal cells can never be blocked."""
        if not self.in_bounds(c) or c in (self.start, self.goal):
            return False
        if c in self.obstacles:
            self.obstacles.discard(c)
        else:
            self.obstacles.add(c)
        return True

    def copy(self) -> "Grid":
        return Grid(self.rows, self.cols, self.start, self.goal, set(self.obstacles))


def random_maze(rows: int, cols: int, density: float, seed: int,
                start: Optional[Cell] = None, goal: Optional[Cell] = None) -> Grid:
    """Random obstacles with the given density (0-0.6); start/goal are never blocked.

    The maze is *not* guaranteed to be solvable on purpose: the UI reports
    unreachable goals honestly instead of hiding them.
    """
    density = float(min(max(density, 0.0), 0.6))
    rng = np.random.default_rng(seed)
    start = start if start is not None else (0, 0)
    goal = goal if goal is not None else (rows - 1, cols - 1)
    mask = rng.random((rows, cols)) < density
    obstacles = {(r, c) for r in range(rows) for c in range(cols) if mask[r, c]}
    obstacles.discard(start)
    obstacles.discard(goal)
    return Grid(rows, cols, start, goal, obstacles)
