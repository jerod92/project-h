"""
FruitCatcherEnvironment — Joystick + Button (multi-appendage).

A basket moves along the bottom of the screen.  Fruits fall from the top at
random positions and speeds.  The agent must:

  1. Steer the basket left/right (Joystick x-axis)
  2. Press the button at the right moment to "catch" a fruit that's above it

Both actions fire every step from the same observation, making this ideal for
training a CompositeGraft with JoystickAppendage + ButtonAppendage.

Single-appendage modes are also supported:
  - Joystick only: basket auto-catches when aligned; just navigate
  - Button only:   basket auto-follows the fruit; just decide when to press

Multi-action expert:
  joystick.x = clamp(dx / speed_scale, -1, 1)  — toward nearest fruit
  button     = 1.0 if any fruit within catch_radius of basket, else 0.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab

_FRUIT_VOCAB = PromptVocab([
    "Move the basket to catch falling fruits and press the button to grab each one.",
    "Steer the basket under a fruit, then press to catch it before it hits the ground.",
    "Navigate the basket beneath falling fruits. Press the button when a fruit is above.",
    "Catch fruits: move the basket left/right with the joystick, press to snatch them.",
    "Control the basket to intercept falling fruits. Press button when aligned with one.",
])

_BG          = (30, 28, 45)
_BASKET_FILL = (200, 160, 80)
_BASKET_RIM  = (230, 190, 100)
_GROUND      = (60, 55, 80)
_TEXT        = (200, 200, 210)
_CATCH_FLASH = (120, 240, 120)

_FRUIT_COLORS = {
    "apple":  (220,  60,  60),
    "orange": (230, 140,  40),
    "grape":  (160,  60, 200),
    "lemon":  (220, 210,  40),
    "lime":   ( 80, 200,  80),
}
_FRUIT_NAMES = list(_FRUIT_COLORS.keys())


@dataclass
class _Fruit:
    x: float
    y: float
    speed: float
    kind: str
    caught: bool = False
    missed: bool = False


class FruitCatcherEnvironment(BaseEnvironment):
    """
    Falling-fruit catcher.  CompositeGraft target: JoystickAppendage + ButtonAppendage.

    expert_action() returns {"joystick": [x, 0.0], "button": 0.0 or 1.0}
    step() accepts either that dict or a plain float (joystick x) in single-appendage mode.
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        n_fruits: int = 2,
        max_steps: int = 80,
        basket_width: int = 50,
        catch_radius: int = 28,
        fruit_speed_range: tuple[float, float] = (1.5, 3.5),
    ):
        self.width = width
        self.height = height
        self.n_fruits = n_fruits
        self._max_steps = max_steps
        self.basket_width = basket_width
        self.catch_radius = catch_radius
        self.fruit_speed_range = fruit_speed_range

        self._basket_x: float = width / 2
        self._fruits: list[_Fruit] = []
        self._score: int = 0
        self._step_count: int = 0
        self._last_catch_flash: int = 0
        self._rng = random.Random()
        self._current_prompt = _FRUIT_VOCAB.sample()

    @property
    def prompt(self) -> str:
        return self._current_prompt

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = random.Random(seed)
        self._basket_x = self.width / 2.0
        self._score = 0
        self._step_count = 0
        self._last_catch_flash = -10
        self._fruits = [self._spawn_fruit() for _ in range(self.n_fruits)]
        self._current_prompt = _FRUIT_VOCAB.sample(seed=seed)
        return self._render()

    def step(self, action) -> EnvStepResult:
        self._step_count += 1

        # Decode action — accept dict (composite) or scalar/list (single-appendage)
        if isinstance(action, dict):
            joy_x = float(action.get("joystick", [0.0])[0] if isinstance(action.get("joystick"), (list, tuple)) else action.get("joystick", 0.0))
            press = float(action.get("button", 0.0))
            if hasattr(press, "__len__"):
                press = float(press[0])
        elif isinstance(action, (list, tuple)):
            joy_x = float(action[0])
            press = 0.0
        else:
            joy_x = float(action)
            press = 0.0

        # Move basket
        speed = 6.0
        self._basket_x = float(np.clip(
            self._basket_x + joy_x * speed,
            self.basket_width // 2,
            self.width - self.basket_width // 2,
        ))

        # Move fruits down
        reward = 0.0
        basket_y = self.height - 28

        for fruit in self._fruits:
            if fruit.caught or fruit.missed:
                continue
            fruit.y += fruit.speed

            # Check catch: button pressed AND fruit close enough to basket
            dx = abs(fruit.x - self._basket_x)
            dy = abs(fruit.y - basket_y)
            in_zone = dx < self.catch_radius and dy < self.catch_radius

            if in_zone and press > 0.5:
                fruit.caught = True
                self._score += 1
                reward += 10.0
                self._last_catch_flash = self._step_count

            elif fruit.y > self.height + 20:
                fruit.missed = True
                reward -= 2.0

        # Respawn caught/missed fruits
        for i, f in enumerate(self._fruits):
            if f.caught or f.missed:
                self._fruits[i] = self._spawn_fruit()

        done = self._step_count >= self._max_steps
        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=done,
            info={
                "score": self._score,
                "step": self._step_count,
                "success": self._score >= max(1, self.n_fruits * (self._max_steps // 20)),
            },
        )

    def expert_action(self) -> dict:
        """Returns {"joystick": [x, 0.0], "button": 0.0_or_1.0}."""
        basket_y = self.height - 28
        # Find nearest active fruit
        active = [f for f in self._fruits if not f.caught and not f.missed]
        if not active:
            return {"joystick": [0.0, 0.0], "button": 0.0}

        nearest = min(active, key=lambda f: abs(f.x - self._basket_x))
        dx = nearest.x - self._basket_x
        joy_x = float(np.clip(dx / 30.0, -1.0, 1.0))

        # Press if any fruit is close
        press = 0.0
        for f in active:
            if abs(f.x - self._basket_x) < self.catch_radius and abs(f.y - basket_y) < self.catch_radius:
                press = 1.0
                break

        return {"joystick": [joy_x, 0.0], "button": press}

    def _spawn_fruit(self) -> _Fruit:
        margin = 20
        x = float(self._rng.uniform(margin, self.width - margin))
        speed = float(self._rng.uniform(*self.fruit_speed_range))
        kind = self._rng.choice(_FRUIT_NAMES)
        return _Fruit(x=x, y=-20.0, speed=speed, kind=kind)

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), _BG)
        draw = ImageDraw.Draw(img)

        # Ground
        draw.rectangle([0, self.height - 12, self.width, self.height], fill=_GROUND)

        # Catch flash
        if self._step_count - self._last_catch_flash < 4:
            draw.rectangle([0, 0, self.width, self.height],
                           outline=_CATCH_FLASH, width=4)

        # Fruits
        fr = 12
        for fruit in self._fruits:
            if fruit.caught or fruit.missed:
                continue
            fx, fy = int(fruit.x), int(fruit.y)
            rgb = _FRUIT_COLORS[fruit.kind]
            draw.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=rgb,
                         outline=tuple(max(0, c - 50) for c in rgb), width=2)

        # Basket
        bx = int(self._basket_x)
        bw = self.basket_width
        by = self.height - 28
        draw.rectangle([bx - bw // 2, by, bx + bw // 2, by + 16], fill=_BASKET_FILL)
        draw.rectangle([bx - bw // 2 - 3, by, bx + bw // 2 + 3, by + 4], fill=_BASKET_RIM)

        # HUD
        draw.text((4, 4), f"score: {self._score}  step: {self._step_count}/{self._max_steps}",
                  fill=_TEXT)

        return img
