"""
Grid World environment — D-pad appendage training.

The agent (red circle) navigates an NxN grid to reach a goal (green circle),
avoiding randomly placed walls (dark cells). Solved optimally by BFS.

Observation: PIL image of the grid, one cell per pixel block.
Action:      DPadButton (STAY=0, UP=1, DOWN=2, LEFT=3, RIGHT=4)
Reward:      +20 at goal, −0.1 per step, −0.5 per wall bump, −2 on timeout.

Expert policy: BFS shortest path — produces perfect demonstrations for BC.

Map generation randomizes walls with a configurable density and guarantees
reachability by carving an L-shaped fallback path if the random map is blocked.
"""

from enum import IntEnum

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab, GRID_WORLD_VOCAB

# Cell types
_EMPTY = 0
_WALL = 1

# Color palette  (R, G, B)
_COLOR_BG = (235, 235, 232)
_COLOR_EMPTY = (230, 228, 225)
_COLOR_WALL = (55, 52, 60)
_COLOR_GOAL_FILL = (45, 200, 105)
_COLOR_GOAL_OUTLINE = (20, 155, 70)
_COLOR_AGENT_FILL = (215, 55, 55)
_COLOR_AGENT_OUTLINE = (155, 20, 20)
_COLOR_GRID = (200, 198, 195)
_COLOR_TEXT = (50, 50, 50)


class GridWorldEnvironment(BaseEnvironment):
    """
    Discrete grid-world navigation environment for D-pad training.

    The grid is rendered as a pixel image where each cell is cell_px × cell_px.
    Walls are dark. The agent (red) must reach the goal (green).
    """

    def __init__(
        self,
        grid_size: int = 8,
        cell_px: int = 28,
        wall_density: float = 0.20,
        max_steps: int | None = None,
    ):
        """
        Args:
            grid_size: Number of cells along each axis.
            cell_px: Pixel size of each grid cell.
            wall_density: Fraction of non-corner cells that become walls.
            max_steps: Episode horizon (defaults to grid_size² * 2).
        """
        self.grid_size = grid_size
        self.cell_px = cell_px
        self.wall_density = wall_density
        self._max_steps = max_steps if max_steps is not None else grid_size * grid_size * 2

        self._grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        self._agent: tuple[int, int] = (0, 0)
        self._goal: tuple[int, int] = (grid_size - 1, grid_size - 1)
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._vocab = PromptVocab(GRID_WORLD_VOCAB)
        self._current_prompt = self._vocab.sample()

    @property
    def prompt(self) -> str:
        return self._current_prompt

    @property
    def image_size(self) -> tuple[int, int]:
        px = self.grid_size * self.cell_px
        return (px, px)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # ------------------------------------------------------------------ #
    #  Core interface                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._current_prompt = self._vocab.sample(seed=seed)
        self._generate_map()
        return self._render()

    def step(self, action) -> EnvStepResult:
        from ..appendages.dpad import DPAD_DELTA, DPadButton

        # Normalize action to DPadButton
        if isinstance(action, DPadButton):
            btn = action
        elif isinstance(action, int):
            btn = DPadButton(action)
        else:
            btn = DPadButton(int(action))

        dx, dy = DPAD_DELTA[btn]  # dx=col delta, dy=row delta
        r, c = self._agent
        nr, nc = r + dy, c + dx

        hit_wall = False
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            if self._grid[nr, nc] == _WALL:
                hit_wall = True  # stay in place
            else:
                self._agent = (nr, nc)
        # out of bounds → stay in place

        self._step_count += 1
        success = self._agent == self._goal
        timeout = self._step_count >= self._max_steps
        done = success or timeout

        reward = -0.1  # time penalty per step
        if hit_wall:
            reward -= 0.5
        if success:
            reward += 20.0
        elif timeout:
            reward -= 2.0

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
            },
        )

    def expert_action(self) -> int:
        """BFS optimal action — always takes the shortest path to the goal."""
        from ..appendages.dpad import DPAD_DELTA, DPadButton

        if self._agent == self._goal:
            return int(DPadButton.STAY)

        queue: list[tuple[tuple[int, int], list[DPadButton]]] = [(self._agent, [])]
        visited: set[tuple[int, int]] = {self._agent}

        while queue:
            pos, path = queue.pop(0)
            for btn in [DPadButton.UP, DPadButton.DOWN, DPadButton.LEFT, DPadButton.RIGHT]:
                dx, dy = DPAD_DELTA[btn]
                nr, nc = pos[0] + dy, pos[1] + dx
                npos = (nr, nc)
                if npos in visited:
                    continue
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue
                if self._grid[nr, nc] == _WALL:
                    continue
                new_path = path + [btn]
                if npos == self._goal:
                    return int(new_path[0])
                visited.add(npos)
                queue.append((npos, new_path))

        return int(DPadButton.STAY)  # unreachable (shouldn't happen after map gen)

    # ------------------------------------------------------------------ #
    #  Map generation                                                       #
    # ------------------------------------------------------------------ #

    def _generate_map(self):
        g = self.grid_size
        grid = np.zeros((g, g), dtype=np.int8)

        # Protected corners (agent spawn, goal, and alternates)
        protected = {(0, 0), (g - 1, g - 1), (0, g - 1), (g - 1, 0)}

        for r in range(g):
            for c in range(g):
                if (r, c) in protected:
                    continue
                if self._rng.random() < self.wall_density:
                    grid[r, c] = _WALL

        self._agent = (0, 0)
        self._goal = (g - 1, g - 1)

        # Guarantee reachability
        if not self._is_reachable(grid, self._agent, self._goal):
            self._carve_path(grid, self._agent, self._goal)

        self._grid = grid

    def _is_reachable(self, grid, start, goal) -> bool:
        visited: set = set()
        stack = [start]
        while stack:
            pos = stack.pop()
            if pos == goal:
                return True
            if pos in visited:
                continue
            visited.add(pos)
            r, c = pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and grid[nr, nc] != _WALL
                ):
                    stack.append((nr, nc))
        return False

    def _carve_path(self, grid, start, goal):
        """Carve an L-shaped open path from start to goal."""
        r_a, c_a = start
        r_g, c_g = goal
        # Horizontal then vertical
        for c in range(min(c_a, c_g), max(c_a, c_g) + 1):
            grid[r_a, c] = _EMPTY
        for r in range(min(r_a, r_g), max(r_a, r_g) + 1):
            grid[r, c_g] = _EMPTY

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _render(self) -> Image.Image:
        px = self.cell_px
        g = self.grid_size
        size = g * px
        img = Image.new("RGB", (size, size), _COLOR_BG)
        draw = ImageDraw.Draw(img)

        # Draw cells
        for r in range(g):
            for c in range(g):
                x0, y0 = c * px, r * px
                x1, y1 = x0 + px - 1, y0 + px - 1
                color = _COLOR_WALL if self._grid[r, c] == _WALL else _COLOR_EMPTY
                draw.rectangle([x0, y0, x1, y1], fill=color)

        # Goal marker (green circle)
        gr, gc = self._goal
        pad = 4
        draw.ellipse(
            [gc * px + pad, gr * px + pad, gc * px + px - pad - 1, gr * px + px - pad - 1],
            fill=_COLOR_GOAL_FILL,
            outline=_COLOR_GOAL_OUTLINE,
            width=2,
        )

        # Agent marker (red circle)
        ar, ac = self._agent
        draw.ellipse(
            [ac * px + pad, ar * px + pad, ac * px + px - pad - 1, ar * px + px - pad - 1],
            fill=_COLOR_AGENT_FILL,
            outline=_COLOR_AGENT_OUTLINE,
            width=2,
        )

        # Grid lines
        for i in range(g + 1):
            draw.line([(i * px, 0), (i * px, size - 1)], fill=_COLOR_GRID, width=1)
            draw.line([(0, i * px), (size - 1, i * px)], fill=_COLOR_GRID, width=1)

        # HUD
        draw.text((2, 2), f"{self._step_count}/{self._max_steps}", fill=_COLOR_TEXT)

        return img
