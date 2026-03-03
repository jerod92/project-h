"""
Maze Navigation environment — D-pad appendage training (advanced).

Generates a **perfect maze** (exactly one path between any two cells) using the
recursive backtracker algorithm (DFS). The agent must navigate from the top-left
entrance to the bottom-right exit.

This is harder than GridWorldEnvironment because:
  - Walls form corridors, not scattered obstacles — dead ends require backtracking
  - The shortest path may require counter-intuitive early moves (go left to reach right)
  - BFS expert still works but the policy needs more steps

Visual:
  - Carved stone / dungeon aesthetic
  - Agent: bright blue circle with direction arrow
  - Exit: glowing green arch

Expert policy: BFS shortest path (always optimal).

Maze representation: internal grid of size (2*rows+1) × (2*cols+1)
  - Odd indices: cell interiors (passable)
  - Even indices: walls or passages between cells
  - DFS carves passages by setting even-index cells to passable
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab, MAZE_VOCAB

# Cell states
_WALL = 0
_OPEN = 1

# Colors
_COLOR_WALL = (45, 38, 50)
_COLOR_FLOOR = (90, 78, 100)
_COLOR_AGENT_FILL = (80, 180, 255)
_COLOR_AGENT_OUTLINE = (30, 120, 200)
_COLOR_EXIT_FILL = (50, 220, 100)
_COLOR_EXIT_OUTLINE = (20, 160, 60)
_COLOR_START_MARK = (200, 100, 60)
_COLOR_TEXT = (200, 190, 210)
_COLOR_BG = (30, 25, 35)


class MazeEnvironment(BaseEnvironment):
    """
    Perfect-maze navigation for D-pad training.

    Every episode generates a new random maze. The agent always starts at
    cell (0,0) and must reach cell (rows-1, cols-1).
    """

    def __init__(
        self,
        rows: int = 7,
        cols: int = 7,
        cell_px: int = 28,
        wall_px: int = 4,
        max_steps: int | None = None,
    ):
        """
        Args:
            rows, cols: Maze dimensions in cells.
            cell_px: Interior size of each cell in pixels.
            wall_px: Thickness of walls in pixels.
            max_steps: Episode horizon (defaults to rows*cols*4).
        """
        self.rows = rows
        self.cols = cols
        self.cell_px = cell_px
        self.wall_px = wall_px
        self._max_steps = max_steps if max_steps is not None else rows * cols * 4

        # Internal grid: (2*rows+1) × (2*cols+1), 1 = open, 0 = wall
        self._grid: np.ndarray = np.zeros((2 * rows + 1, 2 * cols + 1), dtype=np.int8)
        self._agent: tuple[int, int] = (0, 0)   # (row, col) in cell coordinates
        self._goal: tuple[int, int] = (rows - 1, cols - 1)
        self._step_count = 0
        self._visited: set[tuple[int, int]] = set()
        self._rng = np.random.default_rng()

        # Pixel dimensions
        self._pw = cols * (cell_px + wall_px) + wall_px
        self._ph = rows * (cell_px + wall_px) + wall_px
        self._vocab = PromptVocab(MAZE_VOCAB)
        self._current_prompt = self._vocab.sample()

    @property
    def prompt(self) -> str:
        return self._current_prompt

    @property
    def image_size(self) -> tuple[int, int]:
        return (self._pw, self._ph)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # ------------------------------------------------------------------ #
    #  Core interface                                                      #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._agent = (0, 0)
        self._visited = {(0, 0)}
        self._current_prompt = self._vocab.sample(seed=seed)
        self._generate_maze()
        return self._render()

    def step(self, action) -> EnvStepResult:
        from ..appendages.dpad import DPAD_DELTA, DPadButton

        if isinstance(action, DPadButton):
            btn = action
        elif isinstance(action, int):
            btn = DPadButton(action)
        else:
            btn = DPadButton(int(action))

        dx, dy = DPAD_DELTA[btn]   # dx=col delta, dy=row delta
        r, c = self._agent
        nr, nc = r + dy, c + dx

        hit_wall = False
        moved = False

        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            # Check passage in internal grid
            # Cell (r,c) → internal (2r+1, 2c+1)
            # Between (r,c) and (nr,nc) → internal (r+nr+1, c+nc+1)
            ir, ic = r + nr + 1, c + nc + 1
            if self._grid[ir, ic] == _OPEN:
                self._agent = (nr, nc)
                self._visited.add((nr, nc))
                moved = True
            else:
                hit_wall = True
        # else out of bounds → stay

        self._step_count += 1
        success = self._agent == self._goal
        timeout = self._step_count >= self._max_steps
        done = success or timeout

        reward = -0.1  # time cost
        if hit_wall:
            reward -= 0.3
        if moved and self._agent not in self._visited:
            reward += 0.5   # exploration bonus for new cells
        if success:
            reward += 25.0
        elif timeout:
            reward -= 5.0

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=done,
            info={
                "agent": self._agent,
                "goal": self._goal,
                "success": success,
                "steps": self._step_count,
                "hit_wall": hit_wall,
                "cells_visited": len(self._visited),
            },
        )

    def expert_action(self) -> int:
        """BFS shortest path through the maze."""
        from ..appendages.dpad import DPAD_DELTA, DPadButton

        if self._agent == self._goal:
            return int(DPadButton.STAY)

        queue: list[tuple[tuple[int, int], list[DPadButton]]] = [(self._agent, [])]
        visited: set[tuple[int, int]] = {self._agent}

        while queue:
            pos, path = queue.pop(0)
            r, c = pos
            for btn in [DPadButton.UP, DPadButton.DOWN, DPadButton.LEFT, DPadButton.RIGHT]:
                dx, dy = DPAD_DELTA[btn]
                nr, nc = r + dy, c + dx
                if (nr, nc) in visited:
                    continue
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Check passage
                ir, ic = r + nr + 1, c + nc + 1
                if self._grid[ir, ic] != _OPEN:
                    continue
                new_path = path + [btn]
                if (nr, nc) == self._goal:
                    return int(new_path[0])
                visited.add((nr, nc))
                queue.append(((nr, nc), new_path))

        return int(DPadButton.STAY)

    # ------------------------------------------------------------------ #
    #  Maze generation (recursive backtracker / DFS)                      #
    # ------------------------------------------------------------------ #

    def _generate_maze(self):
        grid = np.zeros((2 * self.rows + 1, 2 * self.cols + 1), dtype=np.int8)

        # Open all cell interiors
        for r in range(self.rows):
            for c in range(self.cols):
                grid[2 * r + 1, 2 * c + 1] = _OPEN

        # DFS carve passages
        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))

        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc, dr, dc))

            if neighbors:
                nr, nc, dr, dc = neighbors[self._rng.integers(len(neighbors))]
                # Open wall between (r,c) and (nr,nc)
                grid[2 * r + 1 + dr, 2 * c + 1 + dc] = _OPEN
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        # Open entrance (top-left border) and exit (bottom-right border)
        grid[1, 0] = _OPEN   # entrance on left wall of cell (0,0)
        grid[2 * self.rows - 1, 2 * self.cols] = _OPEN  # exit on right wall

        self._grid = grid

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _cell_rect(self, r: int, c: int) -> tuple[int, int, int, int]:
        """Pixel rect (x0, y0, x1, y1) for cell (r, c)."""
        wp = self.wall_px
        cp = self.cell_px
        x0 = c * (cp + wp) + wp
        y0 = r * (cp + wp) + wp
        return x0, y0, x0 + cp - 1, y0 + cp - 1

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self._pw, self._ph), _COLOR_BG)
        draw = ImageDraw.Draw(img)

        # Draw cell floors and passages by iterating over cells directly.
        # This avoids the coordinate confusion of the internal-grid approach:
        #   Cell (r, c) is at internal grid position (2r+1, 2c+1).
        #   Passage right to (r, c+1) is at internal (2r+1, 2c+2).
        #   Passage down to (r+1, c) is at internal (2r+2, 2c+1).
        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0, x1, y1 = self._cell_rect(r, c)

                # Cell floor
                draw.rectangle([x0, y0, x1, y1], fill=_COLOR_FLOOR)

                # Passage to the right (horizontal corridor)
                if c + 1 < self.cols and self._grid[2 * r + 1, 2 * c + 2] == _OPEN:
                    rx0, ry0, rx1, ry1 = self._cell_rect(r, c + 1)
                    draw.rectangle([x1 + 1, y0, rx0 - 1, ry1], fill=_COLOR_FLOOR)

                # Passage downward (vertical corridor)
                if r + 1 < self.rows and self._grid[2 * r + 2, 2 * c + 1] == _OPEN:
                    dx0, dy0, dx1, dy1 = self._cell_rect(r + 1, c)
                    draw.rectangle([x0, y1 + 1, dx1, dy0 - 1], fill=_COLOR_FLOOR)

        # Start cell marker (dim)
        cp = self.cell_px
        sx0, sy0, sx1, sy1 = self._cell_rect(0, 0)
        pad = cp // 4
        draw.ellipse(
            [sx0 + pad, sy0 + pad, sx1 - pad, sy1 - pad],
            outline=_COLOR_START_MARK,
            width=1,
        )

        # Goal cell (green)
        gr, gc = self._goal
        gx0, gy0, gx1, gy1 = self._cell_rect(gr, gc)
        pad = cp // 5
        draw.ellipse(
            [gx0 + pad, gy0 + pad, gx1 - pad, gy1 - pad],
            fill=_COLOR_EXIT_FILL,
            outline=_COLOR_EXIT_OUTLINE,
            width=2,
        )

        # Agent (blue circle + tiny direction arrow if moving)
        ar, ac = self._agent
        ax0, ay0, ax1, ay1 = self._cell_rect(ar, ac)
        pad = cp // 5
        draw.ellipse(
            [ax0 + pad, ay0 + pad, ax1 - pad, ay1 - pad],
            fill=_COLOR_AGENT_FILL,
            outline=_COLOR_AGENT_OUTLINE,
            width=2,
        )

        # HUD
        draw.text((2, 2), f"step {self._step_count}/{self._max_steps}", fill=_COLOR_TEXT)
        pct = len(self._visited) / (self.rows * self.cols)
        draw.text(
            (2, self._ph - 12),
            f"explored {pct:.0%}",
            fill=_COLOR_TEXT,
        )

        return img
