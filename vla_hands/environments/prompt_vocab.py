"""
Dynamic prompt vocabulary with synonym sets for VLA environments.

Each environment uses a PromptVocab with several equivalent phrasings of its
task description. A new phrasing is sampled on every reset(), so the model
learns to respond to the task regardless of exact wording — improving robustness
and reducing prompt over-fitting.

Usage:
    from .prompt_vocab import PromptVocab, TARGET_NAV_VOCAB

    class MyEnv(BaseEnvironment):
        def __init__(self):
            self._vocab = PromptVocab(TARGET_NAV_VOCAB)

        def reset(self, seed=None):
            self._rng = np.random.default_rng(seed)
            self._current_prompt = self._vocab.sample(seed=seed)
            ...

        @property
        def prompt(self):
            return self._current_prompt
"""

from __future__ import annotations

import random
from typing import Any


class PromptVocab:
    """
    Maintains a list of equivalent prompt templates and samples one per episode.

    Templates may contain {keyword} placeholders — pass them as keyword
    arguments to sample().  The sampled (and formatted) string is stored in
    .current for access between reset() calls.
    """

    def __init__(self, templates: list[str]):
        if not templates:
            raise ValueError("templates must be non-empty")
        self._templates = templates
        self._rng = random.Random()
        self._current: str = templates[0]

    def sample(self, seed: int | None = None, **fmt_kwargs: Any) -> str:
        """
        Pick a random template, format it with any provided kwargs, and
        store the result in .current.

        Args:
            seed: If provided, seeds the internal RNG before sampling
                  (makes resets reproducible when env.reset(seed=N) is used).
            **fmt_kwargs: Values substituted into {placeholder} slots.
        """
        if seed is not None:
            # Derive a prompt-specific seed so it doesn't disturb other RNG state
            self._rng.seed(seed ^ 0xA5C3_F1E2)
        tmpl = self._rng.choice(self._templates)
        self._current = tmpl.format(**fmt_kwargs) if fmt_kwargs else tmpl
        return self._current

    @property
    def current(self) -> str:
        """The most recently sampled prompt."""
        return self._current

    def __len__(self) -> int:
        return len(self._templates)

    def __repr__(self) -> str:
        return (
            f"PromptVocab(n={len(self._templates)}, "
            f"current={self._current[:40]!r}{'...' if len(self._current) > 40 else ''})"
        )


# ── Pre-built vocabulary sets ─────────────────────────────────────────────────

TARGET_NAV_VOCAB: list[str] = [
    "You are controlling a red circle. Navigate it to the blue target marked with X. "
    "Output joystick values to move toward the target.",

    "Move the red agent to the blue X marker using the joystick. "
    "Output (x, y) directional values.",

    "Guide the red dot to the blue cross. Use joystick input to steer.",

    "Steer the red circle toward the blue target. "
    "Positive x moves right, positive y moves down.",

    "The red circle is the agent. Drive it to the blue X. "
    "Emit joystick (x, y) to navigate.",

    "Your agent is the red circle; the goal is the blue X. "
    "Use joystick output to reach the goal as fast as possible.",
]

SPACESHIP_VOCAB: list[str] = [
    "You are piloting a spaceship (white triangle). "
    "Navigate to the gold star using joystick thrust. "
    "Account for your momentum — decelerate before reaching the target.",

    "Pilot the white triangle ship to the gold star. "
    "The joystick controls thrust, not velocity — watch your momentum.",

    "Steer the spaceship to dock with the golden star. "
    "Apply thrust carefully; the ship carries inertia.",

    "Navigate the triangle to the star. "
    "Remember: joystick = acceleration. Brake before arrival.",

    "Guide the white ship to the gold target. "
    "Momentum persists — use counter-thrust to slow down near the star.",

    "Fly the spaceship to the glowing star. "
    "Output thrust (x, y) ∈ [-1, 1]. The ship decelerates slowly.",
]

GRID_WORLD_VOCAB: list[str] = [
    "Navigate the red agent to the green goal on the grid. "
    "Dark cells are walls — avoid them. Use directional buttons to move.",

    "Move the red dot to the green circle. "
    "Black squares are walls; you cannot walk through them.",

    "Steer the red agent to reach the green target. "
    "Use UP, DOWN, LEFT, RIGHT buttons. Walls block movement.",

    "Guide the red marker to the green goal. "
    "Navigate around dark wall cells using the D-pad.",

    "The red agent must reach the green goal. "
    "Press directional buttons to navigate. Walls are impassable.",

    "Get the red circle to the green circle. "
    "Dark tiles are solid walls. Use directional controls.",
]

MAZE_VOCAB: list[str] = [
    "Navigate the blue agent through the maze to the green exit. "
    "Find a path through the corridors — you may need to backtrack.",

    "Guide the blue dot to the green arch at the maze exit. "
    "Use the D-pad to navigate. Dead ends require backtracking.",

    "Move through the dungeon maze to the glowing green exit. "
    "Plan ahead — corridors can be dead ends.",

    "Steer the blue circle to reach the green goal in the maze. "
    "Use directional buttons to explore corridors.",

    "Find your way through the maze to the green exit. "
    "The blue marker is your agent. Backtracking may be necessary.",

    "Navigate the maze corridors with the blue agent. "
    "Reach the green exit. There is exactly one solution path.",
]

BUTTON_PRESS_VOCAB: list[str] = [
    "Look at the circle. Is it {color}? Press the button to answer YES.",

    "Is the circle shown {color}? Activate the button if yes.",

    "Examine the colored circle. Press the button only if it is {color}.",

    "Does the circle match the color {color}? "
    "Press YES (button) if it does; do nothing if it does not.",

    "The target color is {color}. Press the button if the circle matches.",

    "Is this circle {color}? Press to confirm, do not press to deny.",
]

MCQ_VOCAB: list[str] = [
    "{question} Press the button for your answer: "
    "A={a}, B={b}, C={c}, D={d}.",

    "Answer the question: {question} "
    "Choices — A: {a}  B: {b}  C: {c}  D: {d}.",

    "{question} Select the correct answer button. "
    "A={a}, B={b}, C={c}, D={d}.",

    "Multiple choice: {question} "
    "Options: A={a}, B={b}, C={c}, D={d}. Press the matching button.",

    "Choose the right answer for: {question} "
    "A: {a} | B: {b} | C: {c} | D: {d}.",
]

POINTING_VOCAB: list[str] = [
    "Tap the {color} circle on the screen. "
    "Output absolute (x, y) coordinates in [0, 1].",

    "Touch the {color} target. "
    "Provide normalized screen coordinates (x, y) where (0,0) is top-left.",

    "Point to the {color} circle. "
    "Output its position as (x, y) fractions of the image width and height.",

    "Click on the {color} dot. "
    "Return normalized coordinates: x = column / width, y = row / height.",

    "Locate and tap the {color} circle. "
    "Output (x, y) in [0, 1] × [0, 1] where (0, 0) is the top-left corner.",

    "Find the {color} circle and touch it. "
    "Output the normalized tap position (x, y).",
]
