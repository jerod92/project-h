"""
Spaceship Navigation environment — joystick appendage training (momentum variant).

Unlike TargetNavEnvironment (direct position control), this environment uses
Newtonian physics: joystick deflection = **acceleration**, not velocity.
The ship carries momentum between steps and must decelerate to stop at the target.

This is significantly harder for the action head to learn because:
  - The model must predict acceleration, not raw direction
  - Overshooting is penalized (ship keeps moving after joystick released)
  - Optimal trajectories require bang-bang or PD-style control

Visual:
  - Dark space background with subtle star field
  - Agent: small triangle rotated to face its current velocity direction
  - Target: golden 5-pointed star
  - Velocity trail showing recent positions

Expert policy: PD controller
  desired_vel = K_p * (target - pos), clamped to max_speed
  acceleration = K_d * (desired_vel - current_vel), clamped to [-1, 1]
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab, SPACESHIP_VOCAB

_BG_COLOR = (8, 10, 28)
_TRAIL_COLOR = (60, 90, 160)
_SHIP_FILL = (200, 220, 255)
_SHIP_OUTLINE = (100, 150, 255)
_STAR_COLOR = (255, 215, 50)
_STAR_OUTLINE = (200, 160, 20)
_TEXT_COLOR = (120, 140, 200)
_GRID_COLOR = (20, 25, 50)


def _draw_star(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, color, outline):
    """Draw a 5-pointed star centered at (cx, cy) with outer radius r."""
    pts = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5  # start pointing up
        radius = r if i % 2 == 0 else r * 0.42
        pts.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
    draw.polygon(pts, fill=color, outline=outline)


def _draw_triangle(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                   angle_rad: float, size: float, fill, outline):
    """Draw a triangle (ship) pointing in direction angle_rad."""
    # Nose at angle_rad, wings at ±140°
    pts = []
    for da in [0, 2.44, -2.44]:  # 0°, ~140°, ~-140°
        a = angle_rad + da
        wing_r = size if da == 0 else size * 0.55
        pts.append((cx + wing_r * math.cos(a), cy - wing_r * math.sin(a)))
    draw.polygon(pts, fill=fill, outline=outline)


class SpaceshipNavEnvironment(BaseEnvironment):
    """
    Physics-based 2D navigation for joystick training.

    The ship has mass-like inertia: joystick → acceleration, not velocity.
    The agent must predict corrective accelerations to rendezvous with the star.
    """

    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        max_steps: int = 120,
        max_speed: float = 6.0,
        thrust: float = 0.8,
        drag: float = 0.92,
        success_radius: float = 14.0,
        n_bg_stars: int = 40,
    ):
        """
        Args:
            max_speed: Velocity magnitude cap (pixels/step).
            thrust: Pixels/step² per unit joystick deflection.
            drag: Velocity multiplier per step (< 1 = damping).
            success_radius: Distance to target that counts as docking.
            n_bg_stars: Number of decorative background stars.
        """
        self.width = width
        self.height = height
        self._max_steps = max_steps
        self.max_speed = max_speed
        self.thrust = thrust
        self.drag = drag
        self.success_radius = success_radius
        self.n_bg_stars = n_bg_stars

        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._target = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._trail: list[tuple[float, float]] = []
        self._bg_star_positions: list[tuple[float, float]] = []
        self._prev_dist = 0.0
        self._rng = np.random.default_rng()
        self._vocab = PromptVocab(SPACESHIP_VOCAB)
        self._current_prompt = self._vocab.sample()

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
        self._trail.clear()
        self._vel[:] = 0.0
        self._current_prompt = self._vocab.sample(seed=seed)

        margin = 30
        min_sep = 80.0
        for _ in range(200):
            px = self._rng.uniform(margin, self.width - margin)
            py = self._rng.uniform(margin, self.height - margin)
            tx = self._rng.uniform(margin, self.width - margin)
            ty = self._rng.uniform(margin, self.height - margin)
            if math.hypot(tx - px, ty - py) >= min_sep:
                break
        self._pos = np.array([px, py], dtype=np.float32)
        self._target = np.array([tx, ty], dtype=np.float32)

        # Small random initial velocity
        self._vel = self._rng.uniform(-0.5, 0.5, 2).astype(np.float32)

        # Static background stars (fixed per episode for consistency)
        self._bg_star_positions = [
            (float(self._rng.uniform(0, self.width)),
             float(self._rng.uniform(0, self.height)))
            for _ in range(self.n_bg_stars)
        ]
        self._trail.append(tuple(self._pos))
        self._prev_dist = float(np.linalg.norm(self._target - self._pos))

        return self._render()

    def step(self, action) -> EnvStepResult:
        if hasattr(action, "x"):
            ax, ay = float(action.x), float(action.y)
        else:
            ax, ay = float(action[0]), float(action[1])

        ax = float(np.clip(ax, -1.0, 1.0))
        ay = float(np.clip(ay, -1.0, 1.0))

        # Physics update
        self._vel[0] += ax * self.thrust
        self._vel[1] += ay * self.thrust

        # Drag and speed cap
        speed = float(np.linalg.norm(self._vel))
        if speed > self.max_speed:
            self._vel = self._vel * (self.max_speed / speed)
        self._vel *= self.drag

        # Position update with soft wall bounce
        self._pos += self._vel
        for i, bound in enumerate([self.width, self.height]):
            if self._pos[i] < 0:
                self._pos[i] = 0.0
                self._vel[i] *= -0.5
            elif self._pos[i] > bound:
                self._pos[i] = float(bound)
                self._vel[i] *= -0.5

        self._trail.append(tuple(self._pos))
        if len(self._trail) > 30:
            self._trail.pop(0)
        self._step_count += 1

        dist = float(np.linalg.norm(self._target - self._pos))
        reward = (self._prev_dist - dist) * 0.4 - 0.02 * speed
        self._prev_dist = dist

        success = dist <= self.success_radius
        timeout = self._step_count >= self._max_steps
        done = success or timeout

        if success:
            reward += 25.0
        elif timeout:
            reward -= 5.0

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=done,
            info={
                "distance": dist,
                "speed": speed,
                "success": success,
                "steps": self._step_count,
            },
        )

    def expert_action(self) -> tuple[float, float]:
        """
        PD controller: steer toward desired velocity vector.
        desired_vel = Kp * (target - pos), clipped to max_speed
        accel = Kd * (desired_vel - current_vel), clipped to [-1,1]
        """
        delta = self._target - self._pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            # Brake
            braking = -self._vel * 1.5 / max(self.thrust, 1e-6)
            return (float(np.clip(braking[0], -1, 1)), float(np.clip(braking[1], -1, 1)))

        # Desired velocity proportional to distance, capped at max_speed
        kp = min(1.0, dist / 40.0)
        desired_vel = (delta / dist) * self.max_speed * kp
        accel = (desired_vel - self._vel) * 1.5 / max(self.thrust, 1e-6)
        accel = np.clip(accel, -1.0, 1.0)
        return (float(accel[0]), float(accel[1]))

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), _BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Subtle grid lines
        step = 40
        for x in range(0, self.width, step):
            draw.line([(x, 0), (x, self.height)], fill=_GRID_COLOR, width=1)
        for y in range(0, self.height, step):
            draw.line([(0, y), (self.width, y)], fill=_GRID_COLOR, width=1)

        # Background stars (tiny dots)
        for bx, by in self._bg_star_positions:
            r = 1
            draw.ellipse([bx - r, by - r, bx + r, by + r], fill=(200, 210, 240))

        # Velocity trail (fading blue dots)
        n = len(self._trail)
        for i, pos in enumerate(self._trail[:-1]):
            alpha_f = (i + 1) / max(n, 1)
            c = int(30 + 80 * alpha_f)
            r = max(1, int(3 * alpha_f))
            px, py = int(pos[0]), int(pos[1])
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(c, c + 20, c + 60))

        # Target star
        tx, ty = float(self._target[0]), float(self._target[1])
        sr = int(self.success_radius)
        draw.ellipse(
            [tx - sr, ty - sr, tx + sr, ty + sr],
            outline=(100, 80, 20),
            width=1,
        )
        _draw_star(draw, tx, ty, 10.0, _STAR_COLOR, _STAR_OUTLINE)

        # Ship — triangle pointing in velocity direction (or up if nearly still)
        px, py = float(self._pos[0]), float(self._pos[1])
        speed = float(np.linalg.norm(self._vel))
        if speed > 0.1:
            angle = math.atan2(-self._vel[1], self._vel[0])  # y is flipped in pixel coords
        else:
            angle = math.pi / 2  # point up
        _draw_triangle(draw, px, py, angle, 9.0, _SHIP_FILL, _SHIP_OUTLINE)

        # HUD
        vel_mag = float(np.linalg.norm(self._vel))
        draw.text((4, 4), f"step {self._step_count}/{self._max_steps}", fill=_TEXT_COLOR)
        draw.text((4, self.height - 22), f"spd {vel_mag:.1f}", fill=_TEXT_COLOR)
        dist = float(np.linalg.norm(self._target - self._pos))
        draw.text((4, self.height - 12), f"dst {dist:.0f}px", fill=_TEXT_COLOR)

        return img
