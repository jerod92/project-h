"""
Pointing / Tap environment — TouchscreenAppendage training.

The agent must tap one named colored circle among several distractors.
The correct target is specified in the task prompt (via PromptVocab).

This environment trains the TouchscreenAppendage to:
  - Understand the visual layout of the scene
  - Identify the target color in the image
  - Map that visual location to a normalised (x, y) coordinate

Unlike TargetNavEnvironment (which moves an agent over multiple steps),
pointing is a **single-step task**: produce (x, y) → evaluate distance to
target → done.  This mirrors real-world touchscreen use cases like
"tap the red button" or "click on the trash icon."

Observation: PIL image with N colored circles at random positions.
Action:      TouchPoint (x, y) ∈ [0, 1]²  (normalised screen coordinates).
Reward:      Gaussian-shaped: +15 at bull's-eye, falling to 0 at 3× radius.

Expert policy: normalised center of the target circle.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab, POINTING_VOCAB

# ── Color palette ─────────────────────────────────────────────────────────────

_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "red":     (220, 60,  60),
    "green":   (60,  200, 80),
    "blue":    (60,  100, 220),
    "yellow":  (230, 210, 40),
    "cyan":    (40,  200, 210),
    "purple":  (160, 60,  200),
    "orange":  (230, 120, 40),
    "pink":    (230, 100, 160),
}

_ALL_COLOR_NAMES = list(_NAMED_COLORS.keys())

_BG = (245, 245, 242)
_LABEL_TEXT = (50, 50, 50)
_HIT_RING_COLOR = (80, 200, 80)
_MISS_RING_COLOR = (200, 80, 80)
_CURSOR_COLOR = (30, 30, 30)
_TEXT_COLOR = (70, 70, 70)


class PointingEnvironment(BaseEnvironment):
    """
    Single-step pointing / tapping task for TouchscreenAppendage.

    Each episode:
      - Spawns N colored circles at random positions.
      - One circle is designated the target; its color is given in the prompt.
      - The model outputs (x, y) ∈ [0, 1]².
      - Reward is a Gaussian centred on the target circle.

    The episode always lasts exactly one step.
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        n_distractors: int = 3,   # total circles = n_distractors + 1 (target)
        circle_radius: int = 20,
        max_steps: int = 1,
        n_colors: int = 4,        # how many distinct colors to use per episode
    ):
        """
        Args:
            width, height:    Canvas dimensions in pixels.
            n_distractors:    Number of non-target circles.
            circle_radius:    Radius of each circle (pixels).
            max_steps:        Episode steps (keep at 1 for pointing).
            n_colors:         How many different colors appear per episode
                              (≤ len(_ALL_COLOR_NAMES)).
        """
        self.width = width
        self.height = height
        self.n_distractors = n_distractors
        self.circle_radius = circle_radius
        self._max_steps = max_steps
        self.n_colors = min(n_colors, len(_ALL_COLOR_NAMES))

        self._circles: list[dict] = []   # [{"color": str, "x": float, "y": float}]
        self._target_idx: int = 0
        self._step_count: int = 0
        self._last_tap: tuple[float, float] | None = None
        self._rng = np.random.default_rng()
        self._vocab = PromptVocab(POINTING_VOCAB)
        self._current_prompt = self._vocab.sample(color="red")

    # ------------------------------------------------------------------ #
    #  BaseEnvironment interface                                           #
    # ------------------------------------------------------------------ #

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
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_tap = None

        # Choose a random subset of colors for this episode
        color_pool = list(self._rng.choice(
            _ALL_COLOR_NAMES,
            size=self.n_colors,
            replace=False,
        ))

        n_circles = 1 + self.n_distractors
        assigned_colors = [
            color_pool[int(self._rng.integers(len(color_pool)))]
            for _ in range(n_circles)
        ]
        # Ensure at least two distinct colors so the task is not trivial
        if len(set(assigned_colors)) == 1 and len(color_pool) > 1:
            assigned_colors[1] = color_pool[(color_pool.index(assigned_colors[0]) + 1) % len(color_pool)]

        # Place circles with minimum separation
        margin = self.circle_radius + 10
        min_sep = self.circle_radius * 2.8
        positions: list[tuple[float, float]] = []
        for _ in range(n_circles):
            for _attempt in range(300):
                x = float(self._rng.uniform(margin, self.width - margin))
                y = float(self._rng.uniform(margin, self.height - margin))
                if all(math.hypot(x - px, y - py) > min_sep for px, py in positions):
                    positions.append((x, y))
                    break
            else:
                # Fallback: place on a grid
                row = len(positions) // 3
                col = len(positions) % 3
                positions.append((
                    margin + col * (self.width - 2 * margin) / 2,
                    margin + row * (self.height - 2 * margin) / 2,
                ))

        self._circles = [
            {"color": assigned_colors[i], "x": positions[i][0], "y": positions[i][1]}
            for i in range(n_circles)
        ]

        # Pick the target (first circle — shuffle later so it's random)
        self._target_idx = int(self._rng.integers(n_circles))
        target_color = self._circles[self._target_idx]["color"]
        self._current_prompt = self._vocab.sample(seed=seed, color=target_color)

        return self._render()

    def step(self, action) -> EnvStepResult:
        # Accept TouchPoint, [x, y], (x, y), or array
        if hasattr(action, "x"):
            tx, ty = float(action.x), float(action.y)
        elif isinstance(action, (list, tuple)):
            tx, ty = float(action[0]), float(action[1])
        else:
            arr = list(action)
            tx, ty = float(arr[0]), float(arr[1])

        # Clamp to [0, 1]
        tx = float(np.clip(tx, 0.0, 1.0))
        ty = float(np.clip(ty, 0.0, 1.0))
        self._last_tap = (tx, ty)
        self._step_count += 1

        # Convert normalised tap → pixel coords
        tap_px = tx * self.width
        tap_py = ty * self.height

        # Distance to target centre (in pixels)
        tgt = self._circles[self._target_idx]
        dist = math.hypot(tap_px - tgt["x"], tap_py - tgt["y"])

        # Gaussian reward centred on target; decays to ~0 at 3× radius
        sigma = self.circle_radius * 1.5
        reward = 15.0 * math.exp(-0.5 * (dist / sigma) ** 2)

        # Binary success: tap within 1 radius
        success = dist <= self.circle_radius * 1.2

        # Penalty for tapping a distractor circle (if not target)
        for i, c in enumerate(self._circles):
            if i == self._target_idx:
                continue
            d = math.hypot(tap_px - c["x"], tap_py - c["y"])
            if d <= self.circle_radius * 1.2:
                reward -= 3.0
                break

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=True,
            info={
                "tap": (tx, ty),
                "tap_px": (tap_px, tap_py),
                "target_pos": (tgt["x"] / self.width, tgt["y"] / self.height),
                "distance_px": dist,
                "success": success,
                "target_color": tgt["color"],
                "steps": self._step_count,
            },
        )

    def expert_action(self) -> tuple[float, float]:
        """Return the normalised centre of the target circle."""
        tgt = self._circles[self._target_idx]
        return (tgt["x"] / self.width, tgt["y"] / self.height)

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), _BG)
        draw = ImageDraw.Draw(img)

        r = self.circle_radius

        # Draw all circles
        for i, c in enumerate(self._circles):
            cx, cy = int(c["x"]), int(c["y"])
            rgb = _NAMED_COLORS[c["color"]]
            outline_rgb = tuple(max(0, v - 50) for v in rgb)
            is_target = i == self._target_idx

            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=rgb,
                outline=outline_rgb,
                width=3 if is_target else 1,
            )

            # Small color label below each circle
            draw.text(
                (cx - 10, cy + r + 3),
                c["color"][:3].upper(),
                fill=_LABEL_TEXT,
            )

        # Show tap cursor from previous step
        if self._last_tap is not None:
            lx = int(self._last_tap[0] * self.width)
            ly = int(self._last_tap[1] * self.height)
            cr = 6
            tgt = self._circles[self._target_idx]
            dist = math.hypot(lx - tgt["x"], ly - tgt["y"])
            ring_color = _HIT_RING_COLOR if dist <= r * 1.2 else _MISS_RING_COLOR
            draw.ellipse([lx - cr, ly - cr, lx + cr, ly + cr],
                         outline=ring_color, width=2)
            draw.line([(lx - cr - 2, ly), (lx + cr + 2, ly)], fill=_CURSOR_COLOR, width=1)
            draw.line([(lx, ly - cr - 2), (lx, ly + cr + 2)], fill=_CURSOR_COLOR, width=1)

        # HUD
        tgt_color = self._circles[self._target_idx]["color"] if self._circles else "?"
        draw.text((4, 4), f"tap the {tgt_color} circle", fill=_TEXT_COLOR)
        draw.text(
            (4, self.height - 14),
            f"step {self._step_count}/{self._max_steps}",
            fill=_TEXT_COLOR,
        )

        return img
