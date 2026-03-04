"""
MCQNavigatorEnvironment — DPad + MultiButton (multi-appendage combo).

A grid has four colour-coded answer zones (A=red, B=green, C=blue, D=yellow).
A question is shown at the top.  The agent can answer in two ways:

  Mode "navigate"   (DPad only):   Walk into the correct zone — zone entry = answer.
  Mode "button"     (MultiButton): Press the matching lettered button directly.
  Mode "both"       (DPad + MultiButton): Navigate AND button both required for full score.

In "both" mode:
  - Navigating to the correct zone: +8 points
  - Pressing the right button while in the correct zone: +12 more points
  - Pressing the wrong button: −5 points

Questions rotate between: dot counting, arithmetic, colour ID.

expert_action() → {"dpad": int, "multibutton": [0,0,1,0]} in "both" mode
                  OR int (dpad) / list (multibutton) in single modes.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab

_NAV_VOCAB = PromptVocab([
    "{question} Navigate to zone {answer} OR press the matching button.",
    "Answer: {question} Go to zone {answer} (colored region) or press button {answer}.",
    "{question} Move to the {answer} zone or select button {answer}.",
    "Multiple choice: {question} Answer zone {answer} is the target. Or press {answer}.",
    "{question} Either enter zone {answer} or press the {answer} button to score.",
])

_BG       = (240, 238, 232)
_TEXT     = (40,  40,  40)
_QBOX     = (220, 218, 210)

_ZONE_COLORS = {
    "A": (220, 80,  80,  180),   # red (with alpha placeholder — we use fill directly)
    "B": (80,  190, 80,  180),   # green
    "C": (80,  100, 220, 180),   # blue
    "D": (210, 170, 40,  180),   # yellow
}
_ZONE_RGB = {k: v[:3] for k, v in _ZONE_COLORS.items()}
_ZONE_LABELS = ["A", "B", "C", "D"]
_AGENT = (50, 50, 200)


class _Question(NamedTuple):
    text: str
    correct: int   # 0-based index into ABCD


def _dot_question(rng: random.Random) -> _Question:
    n = rng.randint(1, 4)
    text = f"How many dots? (shown as •)"
    return _Question(text=f"Count: {'• ' * n}({n} dots)", correct=n - 1)


def _arithmetic_question(rng: random.Random) -> _Question:
    ops = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3)]
    a, b = rng.choice(ops)
    ans = a + b - 1   # 1-4 range
    ans = max(1, min(4, ans))
    return _Question(text=f"{a} + {b} = ?", correct=ans - 1)


def _color_question(rng: random.Random) -> _Question:
    target = rng.choice(_ZONE_LABELS)
    idx = _ZONE_LABELS.index(target)
    zone_color_names = {"A": "red", "B": "green", "C": "blue", "D": "yellow"}
    return _Question(
        text=f"Which zone is {zone_color_names[target]}?",
        correct=idx,
    )


_Q_GENERATORS = [_dot_question, _arithmetic_question, _color_question]

_DPAD_STAY  = 0
_DPAD_UP    = 1
_DPAD_DOWN  = 2
_DPAD_LEFT  = 3
_DPAD_RIGHT = 4
_DELTA = {_DPAD_STAY:(0,0), _DPAD_UP:(-1,0), _DPAD_DOWN:(1,0), _DPAD_LEFT:(0,-1), _DPAD_RIGHT:(0,1)}


class MCQNavigatorEnvironment(BaseEnvironment):
    """
    Answer MCQ by navigating AND/OR pressing buttons.

    CompositeGraft target: DPadAppendage + MultiButtonAppendage(4).

    expert_action() → {"dpad": int, "multibutton": [float×4]}
    """

    GRID = 5        # 5×5 grid, 4 corner zones each 2×2 cells
    LABELS = _ZONE_LABELS

    def __init__(
        self,
        cell_px: int = 36,
        max_steps: int = 30,
        mode: str = "both",   # "navigate" | "button" | "both"
    ):
        self.cell_px = cell_px
        self._max_steps = max_steps
        self.mode = mode

        g = self.GRID
        self._pw = g * cell_px
        self._ph = g * cell_px + 30

        self._question: _Question | None = None
        self._agent: tuple[int, int] = (g // 2, g // 2)
        self._step_count = 0
        self._nav_scored = False
        self._btn_scored = False
        self._rng = random.Random()
        self._current_prompt = ""

    @property
    def prompt(self) -> str:
        return self._current_prompt

    @property
    def image_size(self) -> tuple[int, int]:
        return (self._pw, self._ph)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # Zone layout (top-left corners in grid coords)
    _ZONE_ORIGINS = {"A": (0, 0), "B": (0, 3), "C": (3, 0), "D": (3, 3)}
    _ZONE_SIZE = 2   # 2×2 cells per zone

    def _zone_at(self, r: int, c: int) -> str | None:
        for label, (zr, zc) in self._ZONE_ORIGINS.items():
            if zr <= r < zr + self._ZONE_SIZE and zc <= c < zc + self._ZONE_SIZE:
                return label
        return None

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = random.Random(seed)
        self._step_count = 0
        self._nav_scored = False
        self._btn_scored = False
        self._agent = (self.GRID // 2, self.GRID // 2)
        self._question = self._rng.choice(_Q_GENERATORS)(self._rng)
        answer_label = _ZONE_LABELS[self._question.correct]
        self._current_prompt = _NAV_VOCAB.sample(
            seed=seed,
            question=self._question.text,
            answer=answer_label,
        )
        return self._render()

    def step(self, action) -> EnvStepResult:
        self._step_count += 1
        q = self._question

        # Decode
        if isinstance(action, dict):
            dpad_val = int(action.get("dpad", 0))
            mb_raw = action.get("multibutton", [0.0] * 4)
            if isinstance(mb_raw, (list, tuple)):
                mb = [float(v) for v in mb_raw]
            else:
                mb = [0.0, 0.0, 0.0, 0.0]
        else:
            # single-appendage fallback
            if isinstance(action, (list, tuple)) and len(action) == 4:
                dpad_val = 0
                mb = [float(v) for v in action]
            else:
                dpad_val = int(action) if not isinstance(action, float) else 0
                mb = [0.0] * 4

        # Move
        dr, dc = _DELTA.get(dpad_val, (0, 0))
        nr, nc = self._agent[0] + dr, self._agent[1] + dc
        if 0 <= nr < self.GRID and 0 <= nc < self.GRID:
            self._agent = (nr, nc)

        reward = 0.0

        # Navigation score
        zone = self._zone_at(*self._agent)
        if zone is not None and not self._nav_scored:
            if zone == _ZONE_LABELS[q.correct]:
                reward += 8.0
                self._nav_scored = True
            else:
                reward -= 2.0   # wrong zone

        # Button score
        chosen_btn = max(range(4), key=lambda i: mb[i]) if any(v > 0.1 for v in mb) else None
        if chosen_btn is not None and not self._btn_scored:
            if chosen_btn == q.correct and (self.mode != "both" or self._nav_scored):
                reward += 12.0
                self._btn_scored = True
            elif chosen_btn != q.correct:
                reward -= 5.0

        done = (self._nav_scored and self._btn_scored) or self._step_count >= self._max_steps

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=done,
            info={
                "nav_scored": self._nav_scored,
                "btn_scored": self._btn_scored,
                "success": self._nav_scored and self._btn_scored,
                "correct_zone": _ZONE_LABELS[q.correct],
            },
        )

    def expert_action(self) -> dict:
        q = self._question
        correct_label = _ZONE_LABELS[q.correct]
        zr, zc = self._ZONE_ORIGINS[correct_label]
        # BFS toward zone centre
        target_r, target_c = zr + self._ZONE_SIZE // 2, zc + self._ZONE_SIZE // 2
        dpad = self._bfs_step(target_r, target_c)

        mb = [0.0] * 4
        mb[q.correct] = 1.0

        return {"dpad": dpad, "multibutton": mb}

    def _bfs_step(self, gr: int, gc: int) -> int:
        ar, ac = self._agent
        if (ar, gc) == (gr, gc):
            return _DPAD_STAY
        visited = {(ar, ac)}
        q: deque = deque([((ar, ac), [])])
        while q:
            (r, c), path = q.popleft()
            for d, (dr, dc) in _DELTA.items():
                if d == _DPAD_STAY:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID and 0 <= nc < self.GRID and (nr, nc) not in visited:
                    new_path = path + [d]
                    if (nr, nc) == (gr, gc):
                        return new_path[0] if new_path else _DPAD_STAY
                    visited.add((nr, nc))
                    q.append(((nr, nc), new_path))
        return _DPAD_STAY

    def _render(self) -> Image.Image:
        cp = self.cell_px
        img = Image.new("RGB", (self._pw, self._ph), _BG)
        draw = ImageDraw.Draw(img)

        # Grid cells & zones
        for r in range(self.GRID):
            for c in range(self.GRID):
                x0, y0 = c * cp, r * cp + 30
                x1, y1 = x0 + cp - 1, y0 + cp - 1
                zone = self._zone_at(r, c)
                if zone:
                    col = _ZONE_RGB[zone]
                    alpha_col = tuple(int(v * 0.6 + 240 * 0.4) for v in col)
                    draw.rectangle([x0, y0, x1, y1], fill=alpha_col, outline=col, width=1)
                    if r == self._ZONE_ORIGINS[zone][0] and c == self._ZONE_ORIGINS[zone][1]:
                        draw.text((x0 + 2, y0 + 2), zone, fill=col)
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=(225, 222, 215), outline=(200, 198, 192))

        # Agent
        ar, ac = self._agent
        ax = ac * cp + cp // 2
        ay = ar * cp + cp // 2 + 30
        ra = cp // 3
        draw.ellipse([ax - ra, ay - ra, ax + ra, ay + ra], fill=_AGENT)

        # Question bar
        q_text = self._question.text if self._question else "?"
        draw.rectangle([0, 0, self._pw, 28], fill=_QBOX)
        draw.text((4, 6), q_text[:50], fill=_TEXT)

        return img
