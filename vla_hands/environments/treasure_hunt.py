"""
TreasureHuntEnvironment — DPad + Button (multi-appendage).

An N×N grid contains hidden treasure tiles (gold stars).  The agent must:

  1. Navigate toward a treasure (DPad: UP/DOWN/LEFT/RIGHT/STAY)
  2. Press the button to *collect* the treasure when standing on it

Collecting all treasures wins the episode.  Pressing the button on an empty
tile is penalised (wasted action).  This trains the model to combine
navigation (DPad) and interaction (Button) into a coherent strategy.

Single-appendage modes:
  - DPad only: pressing button is auto-handled — collecting is automatic on step
  - Button only: agent teleports toward treasure; just decide when to collect

expert_action() → {"dpad": DPadButton, "button": 0.0 or 1.0}
"""

from __future__ import annotations

from collections import deque
from enum import IntEnum

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab

_HUNT_VOCAB = PromptVocab([
    "Navigate to each gold treasure and press the button to collect it.",
    "Find the gold stars on the grid and press the button when you're standing on one.",
    "Use directional controls to reach treasures, then press to pick them up.",
    "Collect all golden treasures: move to each one and press the collect button.",
    "Navigate the grid to find gold stars. Stand on one and press the button to collect.",
])

_BG      = (245, 242, 235)
_WALL    = (60,  55,  50)
_FLOOR   = (220, 215, 205)
_AGENT   = (60,  100, 220)
_TREASURE= (230, 180,  30)
_EMPTY_T = (180, 175, 165)   # already-collected tile
_TEXT    = (50,  50,  50)
_HIT_HL  = (80,  200, 80)
_MISS_HL = (200, 80,  80)


class _DPad(IntEnum):
    STAY  = 0
    UP    = 1
    DOWN  = 2
    LEFT  = 3
    RIGHT = 4


_DELTA = {
    _DPad.STAY: (0, 0),
    _DPad.UP:   (-1, 0),
    _DPad.DOWN: (1, 0),
    _DPad.LEFT: (0, -1),
    _DPad.RIGHT:(0, 1),
}


class TreasureHuntEnvironment(BaseEnvironment):
    """
    Grid-based collect task.  CompositeGraft target: DPadAppendage + ButtonAppendage.

    expert_action() → {"dpad": int (DPadButton value), "button": float}
    step() accepts that dict or a plain int (DPad only).
    """

    def __init__(
        self,
        grid_size: int = 7,
        n_treasures: int = 3,
        n_walls: int = 8,
        cell_px: int = 28,
        max_steps: int = 100,
    ):
        self.grid_size = grid_size
        self.n_treasures = n_treasures
        self.n_walls = n_walls
        self.cell_px = cell_px
        self._max_steps = max_steps
        self._pw = grid_size * cell_px
        self._ph = grid_size * cell_px

        self._grid:  np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int8)
        self._agent: tuple[int, int] = (0, 0)
        self._treasures: set[tuple[int, int]] = set()
        self._collected: set[tuple[int, int]] = set()
        self._step_count = 0
        self._last_action_result: str = ""
        self._rng = np.random.default_rng()
        self._current_prompt = _HUNT_VOCAB.sample()

    @property
    def prompt(self) -> str:
        return self._current_prompt

    @property
    def image_size(self) -> tuple[int, int]:
        return (self._pw, self._ph + 20)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._collected = set()
        self._last_action_result = ""
        self._current_prompt = _HUNT_VOCAB.sample(seed=seed)
        self._generate_map()
        return self._render()

    def _generate_map(self):
        g = self.grid_size
        self._grid[:] = 0
        # Place walls (skip (0,0) and (g-1,g-1))
        all_cells = [(r, c) for r in range(g) for c in range(g)
                     if (r, c) not in {(0, 0)}]
        wall_cells = self._rng.choice(len(all_cells), size=min(self.n_walls, len(all_cells) - self.n_treasures), replace=False)
        for idx in wall_cells:
            r, c = all_cells[int(idx)]
            self._grid[r, c] = 1

        # Place treasures on non-wall, non-start cells
        open_cells = [(r, c) for r in range(g) for c in range(g)
                      if self._grid[r, c] == 0 and (r, c) != (0, 0)]
        t_idxs = self._rng.choice(len(open_cells), size=min(self.n_treasures, len(open_cells)), replace=False)
        self._treasures = {open_cells[int(i)] for i in t_idxs}

        self._agent = (0, 0)
        # Guarantee agent can reach at least one treasure via BFS
        self._ensure_reachable()

    def _ensure_reachable(self):
        """Clear walls on an L-path to the first treasure."""
        if not self._treasures:
            return
        goal = next(iter(self._treasures))
        ar, ac = self._agent
        gr, gc = goal
        for r in range(min(ar, gr), max(ar, gr) + 1):
            self._grid[r, ac] = 0
        for c in range(min(ac, gc), max(ac, gc) + 1):
            self._grid[gr, c] = 0

    def step(self, action) -> EnvStepResult:
        self._step_count += 1

        # Decode
        if isinstance(action, dict):
            dpad_val = int(action.get("dpad", 0))
            press = float(action.get("button", 0.0))
            if hasattr(press, "__len__"):
                press = float(press[0])
        elif hasattr(action, "value"):
            dpad_val = int(action.value)
            press = 0.0
        else:
            dpad_val = int(action)
            press = 0.0

        dpad_val = max(0, min(4, dpad_val))
        dr, dc = _DELTA[_DPad(dpad_val)]
        nr, nc = self._agent[0] + dr, self._agent[1] + dc

        # Move if valid
        reward = 0.0
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self._grid[nr, nc] == 0:
            self._agent = (nr, nc)
            reward += -0.1   # small step cost encourages efficiency
        else:
            reward += -0.2   # wall bump

        # Collect treasure
        pos = self._agent
        on_treasure = pos in self._treasures and pos not in self._collected
        if press > 0.5:
            if on_treasure:
                self._collected.add(pos)
                reward += 15.0
                self._last_action_result = "collect"
            else:
                reward -= 1.0   # wrong press
                self._last_action_result = "miss"
        else:
            self._last_action_result = ""

        all_collected = self._treasures.issubset(self._collected)
        done = all_collected or self._step_count >= self._max_steps

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=done,
            info={
                "collected": len(self._collected),
                "total_treasures": len(self._treasures),
                "success": all_collected,
                "step": self._step_count,
            },
        )

    def expert_action(self) -> dict:
        """BFS toward nearest uncollected treasure, press when on one."""
        remaining = self._treasures - self._collected
        on_treasure = self._agent in remaining

        if not remaining:
            return {"dpad": int(_DPad.STAY), "button": 0.0}

        # BFS to nearest
        goal = self._bfs_nearest(remaining)
        dpad = int(_DPad.STAY)
        if goal:
            ar, ac = self._agent
            gr, gc = goal
            if gr < ar and self._grid[ar - 1, ac] == 0:
                dpad = int(_DPad.UP)
            elif gr > ar and self._grid[ar + 1, ac] == 0:
                dpad = int(_DPad.DOWN)
            elif gc < ac and self._grid[ar, ac - 1] == 0:
                dpad = int(_DPad.LEFT)
            elif gc > ac and self._grid[ar, ac + 1] == 0:
                dpad = int(_DPad.RIGHT)

        return {"dpad": dpad, "button": 1.0 if on_treasure else 0.0}

    def _bfs_nearest(self, targets: set) -> tuple[int, int] | None:
        visited = {self._agent}
        q: deque = deque([(self._agent, [])])
        while q:
            pos, path = q.popleft()
            if pos in targets:
                if path:
                    return path[0]   # first step toward target
                return pos
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = pos[0]+dr, pos[1]+dc
                npos = (nr, nc)
                if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                        and self._grid[nr, nc] == 0 and npos not in visited):
                    visited.add(npos)
                    q.append((npos, path + [npos] if path else [npos]))
        return None

    def _render(self) -> Image.Image:
        cp = self.cell_px
        img = Image.new("RGB", (self._pw, self._ph + 20), _BG)
        draw = ImageDraw.Draw(img)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x0, y0 = c * cp, r * cp
                x1, y1 = x0 + cp - 1, y0 + cp - 1
                if self._grid[r, c] == 1:
                    draw.rectangle([x0, y0, x1, y1], fill=_WALL)
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=_FLOOR)

                pos = (r, c)
                if pos in self._collected:
                    draw.rectangle([x0+4, y0+4, x1-4, y1-4], fill=_EMPTY_T)
                elif pos in self._treasures:
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    s = cp // 4
                    draw.polygon(
                        [(cx, cy-s), (cx+s//2, cy+s//3), (cx-s//2, cy+s//3)],
                        fill=_TREASURE,
                    )

        # Agent
        ar, ac = self._agent
        ax0, ay0 = ac * cp + 3, ar * cp + 3
        ax1, ay1 = ax0 + cp - 7, ay0 + cp - 7
        draw.ellipse([ax0, ay0, ax1, ay1], fill=_AGENT)

        # Flash
        if self._last_action_result == "collect":
            draw.rectangle([0, 0, self._pw, self._ph], outline=_HIT_HL, width=3)
        elif self._last_action_result == "miss":
            draw.rectangle([0, 0, self._pw, self._ph], outline=_MISS_HL, width=2)

        # HUD
        draw.text(
            (2, self._ph + 2),
            f"collected {len(self._collected)}/{len(self._treasures)}  step {self._step_count}",
            fill=_TEXT,
        )
        return img
