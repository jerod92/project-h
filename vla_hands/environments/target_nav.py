"""
Target Navigation environment — joystick appendage training.

The agent (red circle) must navigate to a randomly placed target (blue X).
Both are rendered on a light-gray canvas at configurable resolution.

Observation: PIL image showing agent position, target position, and a motion trail.
Action:      JoystickAction (x, y) ∈ [-1, 1]², mapped to pixel displacement.
Reward:      Proportional to distance improvement per step, +20 on success.

Expert policy: normalize(target - agent_pos), scaled down when near target.
This produces smooth, optimal trajectories suitable for behavioral cloning.

Environment randomization:
  - Agent and target positions are random each episode (minimum separation enforced).
  - Optionally add moving targets or obstacles in future subclasses.
"""

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseEnvironment, EnvStepResult
from .prompt_vocab import PromptVocab, TARGET_NAV_VOCAB

# Color palette
_BG = (245, 245, 242)
_TRAIL = (180, 180, 180)
_AGENT_FILL = (210, 55, 55)
_AGENT_OUTLINE = (150, 20, 20)
_TARGET_FILL = (40, 110, 220)
_TARGET_OUTLINE = (20, 70, 170)
_SUCCESS_RING = (80, 160, 230)
_TEXT = (80, 80, 80)


class TargetNavEnvironment(BaseEnvironment):
    """
    Continuous 2D target-navigation environment for joystick training.

    The scene is rendered as a (width × height) pixel image that a VLM
    can parse visually: a red circle (agent) chasing a blue X (target).
    """

    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        agent_radius: int = 8,
        target_radius: int = 8,
        max_steps: int = 100,
        speed: float = 8.0,
        success_radius: float = 14.0,
        trail_length: int = 25,
    ):
        """
        Args:
            width, height: Image dimensions in pixels.
            agent_radius: Radius of the agent circle in pixels.
            target_radius: Radius of the target marker in pixels.
            max_steps: Episode horizon.
            speed: Pixels moved per unit joystick deflection.
            success_radius: Distance (px) at which the episode is won.
            trail_length: Number of past agent positions to display.
        """
        self.width = width
        self.height = height
        self.agent_radius = agent_radius
        self.target_radius = target_radius
        self._max_steps = max_steps
        self.speed = speed
        self.success_radius = success_radius
        self.trail_length = trail_length

        self._agent_pos = np.zeros(2, dtype=np.float32)
        self._target_pos = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._trail: list[tuple[float, float]] = []
        self._prev_dist = 0.0
        self._rng = np.random.default_rng()
        self._vocab = PromptVocab(TARGET_NAV_VOCAB)
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
        self._current_prompt = self._vocab.sample(seed=seed)

        margin = max(self.agent_radius + 4, 20)
        min_sep = max(self.success_radius * 3, 50.0)

        # Place agent and target with minimum separation
        for _ in range(200):
            ax = self._rng.uniform(margin, self.width - margin)
            ay = self._rng.uniform(margin, self.height - margin)
            tx = self._rng.uniform(margin, self.width - margin)
            ty = self._rng.uniform(margin, self.height - margin)
            if np.hypot(tx - ax, ty - ay) >= min_sep:
                break

        self._agent_pos = np.array([ax, ay], dtype=np.float32)
        self._target_pos = np.array([tx, ty], dtype=np.float32)
        self._prev_dist = float(np.linalg.norm(self._target_pos - self._agent_pos))
        self._trail.append((ax, ay))

        return self._render()

    def step(self, action) -> EnvStepResult:
        # Accept JoystickAction, (x, y) tuple, or [x, y] list/array
        if hasattr(action, "x"):
            dx, dy = float(action.x), float(action.y)
        else:
            dx, dy = float(action[0]), float(action[1])

        dx = float(np.clip(dx, -1.0, 1.0))
        dy = float(np.clip(dy, -1.0, 1.0))

        # Move agent, clamped to canvas bounds
        self._agent_pos[0] = float(
            np.clip(self._agent_pos[0] + dx * self.speed, 0, self.width)
        )
        self._agent_pos[1] = float(
            np.clip(self._agent_pos[1] + dy * self.speed, 0, self.height)
        )

        self._trail.append(tuple(self._agent_pos))
        if len(self._trail) > self.trail_length:
            self._trail.pop(0)

        self._step_count += 1
        dist = float(np.linalg.norm(self._target_pos - self._agent_pos))

        # Reward: distance improvement + small control cost
        reward = (self._prev_dist - dist) * 0.5 - 0.01 * float(
            np.hypot(dx, dy)
        )
        self._prev_dist = dist

        success = dist <= self.success_radius
        timeout = self._step_count >= self._max_steps
        done = success or timeout

        if success:
            reward += 20.0
        elif timeout:
            reward -= 3.0

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=done,
            info={
                "distance": dist,
                "success": success,
                "steps": self._step_count,
                "agent_pos": tuple(self._agent_pos.tolist()),
                "target_pos": tuple(self._target_pos.tolist()),
            },
        )

    def expert_action(self) -> tuple[float, float]:
        """
        Optimal joystick: point directly at target, scaled down near goal.
        """
        delta = self._target_pos - self._agent_pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return (0.0, 0.0)
        direction = delta / dist
        # Slow down when close to prevent overshooting
        scale = min(1.0, dist / (self.speed * 2.0))
        return (float(direction[0] * scale), float(direction[1] * scale))

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), _BG)
        draw = ImageDraw.Draw(img)

        # Trail (fading gray dots)
        n_trail = len(self._trail)
        for i, pos in enumerate(self._trail):
            alpha = int(40 + 120 * (i / max(n_trail - 1, 1)))
            r = max(1, self.agent_radius - 5)
            px, py = int(pos[0]), int(pos[1])
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=(alpha, alpha, alpha),
            )

        # Success radius ring around target
        tx, ty = int(self._target_pos[0]), int(self._target_pos[1])
        sr = int(self.success_radius)
        draw.ellipse(
            [tx - sr, ty - sr, tx + sr, ty + sr],
            outline=_SUCCESS_RING,
            width=1,
        )

        # Target: filled circle + X cross
        r = self.target_radius
        draw.ellipse(
            [tx - r, ty - r, tx + r, ty + r],
            fill=_TARGET_FILL,
            outline=_TARGET_OUTLINE,
            width=2,
        )
        pad = 3
        draw.line([tx - r + pad, ty - r + pad, tx + r - pad, ty + r - pad],
                  fill=(255, 255, 255), width=2)
        draw.line([tx + r - pad, ty - r + pad, tx - r + pad, ty + r - pad],
                  fill=(255, 255, 255), width=2)

        # Agent: filled red circle
        ax, ay = int(self._agent_pos[0]), int(self._agent_pos[1])
        r = self.agent_radius
        draw.ellipse(
            [ax - r, ay - r, ax + r, ay + r],
            fill=_AGENT_FILL,
            outline=_AGENT_OUTLINE,
            width=2,
        )

        # HUD
        draw.text(
            (4, 4),
            f"step {self._step_count}/{self._max_steps}",
            fill=_TEXT,
        )
        dist = float(np.linalg.norm(self._target_pos - self._agent_pos))
        draw.text((4, self.height - 14), f"dist {dist:.0f}px", fill=_TEXT)

        return img
