"""
PaintCanvasEnvironment — Touchscreen (sequential waypoint tracing).

A dashed path is drawn on a blank canvas.  The agent must tap each waypoint
in order, tracing the path one touch at a time.  This is a multi-step
touchscreen task: each episode has N waypoints and lasts exactly N steps.

This teaches the TouchscreenAppendage to:
  - Identify the *next* target point among several options
  - Follow a shown sequence rather than acting on a static target

Shapes include: zigzag, spiral steps, L-shape, S-curve, star points.

expert_action() → (x, y) normalised position of the next waypoint.
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab

_PAINT_VOCAB = PromptVocab([
    "Tap each circle on the dashed path in order from start to finish.",
    "Follow the path: tap the next highlighted waypoint in the sequence.",
    "Trace the shape by tapping each dot in order. Start from the circled point.",
    "Touch each waypoint along the dashed line, starting from the first one.",
    "Complete the path by tapping each circle sequentially from left to right.",
    "Tap the waypoints in order to draw the path. The current target is highlighted.",
])

_BG          = (250, 248, 244)
_PATH_DASH   = (180, 175, 170)
_WP_DONE     = (160, 220, 160)   # completed waypoints
_WP_CURRENT  = (60,  120, 220)   # next to tap
_WP_FUTURE   = (200, 195, 190)   # not yet reached
_TAP_HIT     = (80,  200,  80)
_TAP_MISS    = (200,  80,  80)
_TEXT        = (80,   80,  80)
_WP_RADIUS   = 12


def _zigzag(n: int, w: int, h: int) -> list[tuple[float, float]]:
    pts = []
    margin = 30
    for i in range(n):
        x = margin + (w - 2 * margin) * i / max(1, n - 1)
        y = (h // 2 - 40) if i % 2 == 0 else (h // 2 + 40)
        pts.append((x, y))
    return pts


def _lshape(n: int, w: int, h: int) -> list[tuple[float, float]]:
    margin = 30
    half = n // 2
    pts = []
    for i in range(half):
        pts.append((margin + (w // 2 - margin) * i / max(1, half - 1), h - margin))
    for i in range(n - half):
        pts.append((w // 2, h - margin - (h - 2 * margin) * i / max(1, n - half - 1)))
    return pts


def _star_points(n: int, w: int, h: int) -> list[tuple[float, float]]:
    cx, cy, r = w / 2, h / 2, min(w, h) * 0.38
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    # star pattern: alternate outer/inner radius
    pts = []
    for i, a in enumerate(angles):
        rad = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return pts


def _scurve(n: int, w: int, h: int) -> list[tuple[float, float]]:
    margin = 30
    pts = []
    for i in range(n):
        t = i / max(1, n - 1)
        x = margin + (w - 2 * margin) * t
        y = h / 2 + math.sin(t * 2 * math.pi) * (h / 4 - margin)
        pts.append((x, y))
    return pts


_SHAPE_GEN = [_zigzag, _lshape, _star_points, _scurve]


class PaintCanvasEnvironment(BaseEnvironment):
    """
    Sequential tap-to-trace environment.  Target appendage: TouchscreenAppendage.

    Each episode presents a different path shape with N waypoints.
    The agent taps them one by one.  Max_steps == n_waypoints.

    expert_action() → (x, y) as normalised floats ∈ [0, 1]²
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        n_waypoints: int = 8,
        hit_radius_px: int = 20,
    ):
        self.width = width
        self.height = height
        self.n_waypoints = n_waypoints
        self.hit_radius_px = hit_radius_px

        self._waypoints: list[tuple[float, float]] = []
        self._current_wp: int = 0
        self._step_count: int = 0
        self._last_tap: tuple[float, float] | None = None
        self._last_hit: bool = False
        self._rng = random.Random()
        self._current_prompt = _PAINT_VOCAB.sample()

    @property
    def prompt(self) -> str:
        return self._current_prompt

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def max_steps(self) -> int:
        return self.n_waypoints  # one step per waypoint

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = random.Random(seed)
        self._step_count = 0
        self._current_wp = 0
        self._last_tap = None
        self._last_hit = False
        self._current_prompt = _PAINT_VOCAB.sample(seed=seed)

        shape_fn = self._rng.choice(_SHAPE_GEN)
        raw = shape_fn(self.n_waypoints, self.width - 20, self.height - 20)
        # offset so they fit within the canvas
        self._waypoints = [(x + 10, y + 10) for x, y in raw]
        return self._render()

    def step(self, action) -> EnvStepResult:
        self._step_count += 1

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

        target = self._waypoints[self._current_wp]
        dist = math.hypot(tap_px - target[0], tap_py - target[1])
        hit = dist <= self.hit_radius_px * 1.5
        self._last_hit = hit

        if hit:
            reward = 10.0 * max(0.0, 1.0 - dist / (self.hit_radius_px * 2))
            self._current_wp += 1
        else:
            reward = -2.0

        done = self._current_wp >= len(self._waypoints) or self._step_count >= self.max_steps

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=done,
            info={
                "hit": hit,
                "dist_px": dist,
                "waypoints_done": self._current_wp,
                "success": self._current_wp >= len(self._waypoints),
            },
        )

    def expert_action(self) -> tuple[float, float]:
        if self._current_wp >= len(self._waypoints):
            return (0.5, 0.5)
        wp = self._waypoints[self._current_wp]
        return (wp[0] / self.width, wp[1] / self.height)

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), _BG)
        draw = ImageDraw.Draw(img)

        # Draw dashed path between waypoints
        wps = self._waypoints
        for i in range(len(wps) - 1):
            x0, y0 = int(wps[i][0]), int(wps[i][1])
            x1, y1 = int(wps[i + 1][0]), int(wps[i + 1][1])
            # Dashed line
            dist = math.hypot(x1 - x0, y1 - y0)
            n_dashes = max(1, int(dist // 8))
            for d in range(n_dashes):
                t0 = d / n_dashes
                t1 = (d + 0.5) / n_dashes
                px0 = int(x0 + (x1 - x0) * t0)
                py0 = int(y0 + (y1 - y0) * t0)
                px1 = int(x0 + (x1 - x0) * t1)
                py1 = int(y0 + (y1 - y0) * t1)
                draw.line([(px0, py0), (px1, py1)], fill=_PATH_DASH, width=2)

        # Draw waypoints
        r = _WP_RADIUS
        for i, (wx, wy) in enumerate(wps):
            ix, iy = int(wx), int(wy)
            if i < self._current_wp:
                col = _WP_DONE
            elif i == self._current_wp:
                col = _WP_CURRENT
            else:
                col = _WP_FUTURE
            draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=col,
                         outline=(max(0, col[0]-60), max(0, col[1]-60), max(0, col[2]-60)),
                         width=2)
            draw.text((ix - 4, iy - 5), str(i + 1), fill=(255, 255, 255))

        # Last tap marker
        if self._last_tap is not None:
            lx = int(self._last_tap[0] * self.width)
            ly = int(self._last_tap[1] * self.height)
            tc = _TAP_HIT if self._last_hit else _TAP_MISS
            draw.ellipse([lx - 5, ly - 5, lx + 5, ly + 5], outline=tc, width=2)

        draw.text((4, 4), f"tap {self._current_wp+1}/{len(wps)}", fill=_TEXT)
        return img
