"""
WhackAMoleEnvironment — Touchscreen only (spatial reaction task).

Moles pop up at random positions wearing different coloured hats.
The agent taps a mole to whack it.  Points depend on the hat colour:

  Brown mole  →  +5   (common)
  Golden mole →  +15  (rare, only one at a time)
  Bomb        →  −10  (don't tap!)

Each step, moles shift positions slightly or new moles appear.
This trains the TouchscreenAppendage to:
  - Identify high-value targets vs. bombs by colour
  - Produce precise tap coordinates under a changing scene

expert_action() → (x, y) normalised position of the highest-value non-bomb mole.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab

_WAM_VOCAB = PromptVocab([
    "Tap the golden mole for bonus points, or any brown mole. Avoid the bombs!",
    "Whack the moles! Tap the gold one first if you see it. Never tap a bomb.",
    "Hit the moles with your tap. Gold moles score most. Bombs lose points.",
    "Tap a mole to whack it. Gold hat = big bonus, dark bomb = penalty.",
    "Find and tap the moles. Prioritise the golden mole. Avoid black bombs.",
])

_BG         = (100, 160, 80)    # green field
_HOLE       = (60,  40,  20)    # dark brown hole
_MOLE_BODY  = (160, 110, 70)
_HAT_BROWN  = (100,  60,  30)
_HAT_GOLD   = (230, 180,  30)
_HAT_BOMB   = (30,   30,  30)
_EYES       = (240, 240, 230)
_HIT_RING   = (255, 255,  80)
_MISS_RING  = (255,  80,  80)
_TEXT       = (240, 240, 230)


@dataclass
class _Mole:
    x: float          # pixel x centre
    y: float          # pixel y centre
    kind: str         # "brown" | "gold" | "bomb"
    alive: bool = True


def _mole_value(kind: str) -> int:
    return {"brown": 5, "gold": 15, "bomb": -10}[kind]


class WhackAMoleEnvironment(BaseEnvironment):
    """
    Fast spatial reaction game.  Target appendage: TouchscreenAppendage.

    expert_action() → (x, y) normalised tap toward the best active mole.
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        n_moles: int = 4,
        max_steps: int = 20,
        hit_radius_px: int = 22,
        p_gold: float = 0.20,
        p_bomb: float = 0.20,
    ):
        self.width = width
        self.height = height
        self.n_moles = n_moles
        self._max_steps = max_steps
        self.hit_radius_px = hit_radius_px
        self.p_gold = p_gold
        self.p_bomb = p_bomb

        self._moles: list[_Mole] = []
        self._score: int = 0
        self._step_count: int = 0
        self._last_tap: tuple[float, float] | None = None
        self._last_hit_kind: str | None = None
        self._rng = random.Random()
        self._current_prompt = _WAM_VOCAB.sample()

    @property
    def prompt(self) -> str:
        return self._current_prompt

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = random.Random(seed)
        self._step_count = 0
        self._score = 0
        self._last_tap = None
        self._last_hit_kind = None
        self._current_prompt = _WAM_VOCAB.sample(seed=seed)
        self._spawn_moles()
        return self._render()

    def _spawn_moles(self):
        margin = 35
        min_sep = self.hit_radius_px * 2.5
        positions = []
        for _ in range(self.n_moles):
            for _ in range(200):
                x = float(self._rng.uniform(margin, self.width - margin))
                y = float(self._rng.uniform(margin + 20, self.height - margin))
                if all(math.hypot(x - px, y - py) > min_sep for px, py in positions):
                    positions.append((x, y))
                    break
            else:
                positions.append((
                    margin + self._rng.random() * (self.width - 2 * margin),
                    margin + self._rng.random() * (self.height - 2 * margin),
                ))

        # Assign kinds: guarantee at most one gold
        has_gold = False
        moles = []
        for x, y in positions:
            r = self._rng.random()
            if r < self.p_gold and not has_gold:
                kind = "gold"
                has_gold = True
            elif r < self.p_gold + self.p_bomb:
                kind = "bomb"
            else:
                kind = "brown"
            moles.append(_Mole(x=x, y=y, kind=kind))

        self._moles = moles

    def step(self, action) -> EnvStepResult:
        self._step_count += 1
        self._last_hit_kind = None

        if hasattr(action, "x"):
            tx, ty = float(action.x), float(action.y)
        elif isinstance(action, (list, tuple)):
            tx, ty = float(action[0]), float(action[1])
        else:
            arr = list(action)
            tx, ty = float(arr[0]), float(arr[1])

        tx, ty = float(np.clip(tx, 0, 1)), float(np.clip(ty, 0, 1))
        self._last_tap = (tx, ty)
        tap_px = tx * self.width
        tap_py = ty * self.height

        # Check which mole was hit
        reward = -0.5   # small cost for missing
        for mole in self._moles:
            if not mole.alive:
                continue
            dist = math.hypot(tap_px - mole.x, tap_py - mole.y)
            if dist <= self.hit_radius_px * 1.2:
                val = _mole_value(mole.kind)
                reward = float(val)
                self._score += val
                mole.alive = False
                self._last_hit_kind = mole.kind
                break

        # Respawn dead moles
        for i, mole in enumerate(self._moles):
            if not mole.alive:
                self._moles[i] = self._random_mole(avoid=[m for m in self._moles if m.alive])

        # Ensure at most one gold
        gold_count = sum(1 for m in self._moles if m.alive and m.kind == "gold")
        if gold_count > 1:
            for m in self._moles:
                if m.alive and m.kind == "gold":
                    m.kind = "brown"
                    gold_count -= 1
                    if gold_count == 1:
                        break

        done = self._step_count >= self._max_steps

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=done,
            info={
                "score": self._score,
                "hit_kind": self._last_hit_kind,
                "success": self._score > 0,
            },
        )

    def expert_action(self) -> tuple[float, float]:
        """Tap the highest-value alive mole."""
        active = [m for m in self._moles if m.alive]
        if not active:
            return (0.5, 0.5)
        best = max(active, key=lambda m: _mole_value(m.kind))
        return (best.x / self.width, best.y / self.height)

    def _random_mole(self, avoid: list[_Mole]) -> _Mole:
        margin = 35
        for _ in range(100):
            x = float(self._rng.uniform(margin, self.width - margin))
            y = float(self._rng.uniform(margin + 20, self.height - margin))
            if all(math.hypot(x - m.x, y - m.y) > self.hit_radius_px * 2.2 for m in avoid):
                break
        r = self._rng.random()
        has_gold = any(m.kind == "gold" for m in avoid if m.alive)
        if r < self.p_gold and not has_gold:
            kind = "gold"
        elif r < self.p_gold + self.p_bomb:
            kind = "bomb"
        else:
            kind = "brown"
        return _Mole(x=x, y=y, kind=kind)

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), _BG)
        draw = ImageDraw.Draw(img)

        # Draw holes then moles
        r_body = self.hit_radius_px
        for mole in self._moles:
            if not mole.alive:
                continue
            mx, my = int(mole.x), int(mole.y)
            # Hole
            draw.ellipse([mx - r_body, my + r_body // 2,
                          mx + r_body, my + r_body * 2], fill=_HOLE)
            # Body
            draw.ellipse([mx - r_body + 4, my - r_body // 2,
                          mx + r_body - 4, my + r_body], fill=_MOLE_BODY)

            # Hat
            hat_col = {"brown": _HAT_BROWN, "gold": _HAT_GOLD, "bomb": _HAT_BOMB}[mole.kind]
            draw.ellipse([mx - r_body // 2, my - r_body,
                          mx + r_body // 2, my - r_body // 4], fill=hat_col)
            draw.rectangle([mx - r_body // 3, my - r_body - 10,
                            mx + r_body // 3, my - r_body + 2], fill=hat_col)

            # Eyes (skip for bombs)
            if mole.kind != "bomb":
                eo = r_body // 4
                draw.ellipse([mx - eo - 3, my - eo - 2, mx - eo + 3, my - eo + 4], fill=_EYES)
                draw.ellipse([mx + eo - 3, my - eo - 2, mx + eo + 3, my - eo + 4], fill=_EYES)
            else:
                # Skull on bomb
                draw.text((mx - 6, my - 8), "💀", fill=(200, 200, 200))

        # Last tap indicator
        if self._last_tap is not None:
            lx = int(self._last_tap[0] * self.width)
            ly = int(self._last_tap[1] * self.height)
            if self._last_hit_kind == "bomb":
                rc = _MISS_RING
            elif self._last_hit_kind is not None:
                rc = _HIT_RING
            else:
                rc = _MISS_RING
            draw.ellipse([lx - 8, ly - 8, lx + 8, ly + 8], outline=rc, width=2)

        # HUD
        draw.text((4, 4), f"score: {self._score}", fill=_TEXT)
        draw.text((4, self.height - 14), f"step {self._step_count}/{self._max_steps}", fill=_TEXT)

        return img
