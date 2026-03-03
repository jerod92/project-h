"""
Button-press training environments.

Two environments, each calibrating a different button skill:

ButtonPressEnvironment (single ButtonAppendage):
    Shows a colored circle.  The model must press the button if and only if
    the circle matches the target color.  Tests: visual color recognition +
    binary decision → action mapping.

    Expert: deterministic — press iff circle_color == target_color.

    Curriculum difficulty:
      easy   — target color is one of {red, green, blue} (very distinct)
      medium — target includes yellow, cyan, purple
      hard   — target includes near-identical hues (orange vs red, etc.)

MCQButtonEnvironment (MultiButtonAppendage, n_buttons=4):
    Shows a simple visual question with 4 answer choices (A/B/C/D) as a
    rendered text-image.  One button corresponds to each choice.
    The correct answer changes each episode.

    Tasks include:
      - Count the dots (1–4 dots, answer is the count)
      - Pick the color (which circle is red/blue/green?)
      - Simple arithmetic (displayed as text)

    Expert: always selects the correct button (one-hot).

    Reward: +15 correct, −2 wrong, no partial credit.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import BaseEnvironment, EnvStepResult

# ── Color palette shared between environments ─────────────────────────────────

_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "red":    (220, 50,  50),
    "green":  (50,  200, 80),
    "blue":   (50,  80,  220),
    "yellow": (230, 210, 30),
    "cyan":   (30,  200, 210),
    "purple": (160, 50,  200),
    "orange": (230, 120, 30),
    "white":  (230, 230, 230),
}

_EASY_COLORS = ["red", "green", "blue"]
_MEDIUM_COLORS = _EASY_COLORS + ["yellow", "cyan", "purple"]
_HARD_COLORS = _MEDIUM_COLORS + ["orange", "white"]

_BG = (245, 245, 242)
_TEXT_DARK = (40, 40, 40)
_TEXT_MED = (100, 100, 100)
_PANEL_BG = (230, 228, 225)
_CORRECT_HL = (60, 200, 80)
_WRONG_HL = (200, 60, 60)


def _render_text_line(draw: ImageDraw.ImageDraw, text: str, xy: tuple, size: int = 14,
                       fill=(40, 40, 40)):
    """Draw a text string. Falls back gracefully if font size unsupported."""
    try:
        draw.text(xy, text, fill=fill)
    except Exception:
        draw.text(xy, text, fill=fill)


# ── ButtonPressEnvironment ────────────────────────────────────────────────────

class ButtonPressEnvironment(BaseEnvironment):
    """
    Visual color-matching task for single ButtonAppendage.

    Each episode:
      - Chooses a random target color (shown as a label)
      - Renders a circle in a random color
      - Model must press the button iff circle == target color

    The episode is one step long (single decision per observation).
    The environment resets automatically after each correct/incorrect press.
    """

    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        difficulty: str = "easy",   # "easy" | "medium" | "hard"
        max_steps: int = 1,
    ):
        """
        Args:
            difficulty: Controls the color palette size.
            max_steps: Steps per episode (1 = single-shot decision).
        """
        self.width = width
        self.height = height
        self._max_steps = max_steps

        if difficulty == "easy":
            self._color_pool = _EASY_COLORS
        elif difficulty == "medium":
            self._color_pool = _MEDIUM_COLORS
        else:
            self._color_pool = _HARD_COLORS

        # Episode state
        self._circle_color_name: str = "red"
        self._target_color_name: str = "red"
        self._correct_press: bool = True
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._last_result: str = ""  # "correct" | "wrong" | ""

    @property
    def prompt(self) -> str:
        return (
            f"Look at the circle. Is it {self._target_color_name}? "
            "Press the button to answer YES."
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def should_press(self) -> bool:
        """True if the correct answer is to press the button."""
        return self._circle_color_name == self._target_color_name

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_result = ""

        colors = self._color_pool
        self._circle_color_name = colors[int(self._rng.integers(len(colors)))]

        # 50% chance the target matches the circle
        if self._rng.random() < 0.5:
            self._target_color_name = self._circle_color_name
        else:
            others = [c for c in colors if c != self._circle_color_name]
            if others:
                self._target_color_name = others[int(self._rng.integers(len(others)))]
            else:
                self._target_color_name = self._circle_color_name

        return self._render()

    def step(self, action) -> EnvStepResult:
        # action is a float [0,1] or ButtonState
        if hasattr(action, "pressed"):
            pressed = action.pressed
            confidence = action.confidence
        elif isinstance(action, (list, np.ndarray)):
            confidence = float(action[0])
            pressed = confidence > 0.5
        else:
            confidence = float(action)
            pressed = confidence > 0.5

        correct = pressed == self.should_press
        self._step_count += 1

        if correct:
            reward = 10.0 if pressed else 5.0  # pressing when right > abstaining when right
            self._last_result = "correct"
        else:
            reward = -5.0
            self._last_result = "wrong"

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=True,  # single-step episode
            info={
                "pressed": pressed,
                "confidence": confidence,
                "should_press": self.should_press,
                "correct": correct,
                "circle_color": self._circle_color_name,
                "target_color": self._target_color_name,
                "success": correct,
            },
        )

    def expert_action(self) -> float:
        """Return 1.0 if should press, 0.0 if not."""
        return 1.0 if self.should_press else 0.0

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), _BG)
        draw = ImageDraw.Draw(img)

        cx, cy = self.width // 2, self.height // 2 - 20
        r = min(self.width, self.height) // 5

        circle_rgb = _NAMED_COLORS[self._circle_color_name]
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=circle_rgb,
            outline=(max(0, circle_rgb[0] - 40), max(0, circle_rgb[1] - 40),
                     max(0, circle_rgb[2] - 40)),
            width=3,
        )

        # Question text
        question = f"Is this circle {self._target_color_name.upper()}?"
        draw.text((self.width // 2 - 80, self.height - 60), question, fill=_TEXT_DARK)
        draw.text((self.width // 2 - 50, self.height - 44), "Press button = YES", fill=_TEXT_MED)

        # Target color swatch
        swatch_w, swatch_h = 30, 18
        sx = self.width // 2 + 60
        sy = self.height - 55
        target_rgb = _NAMED_COLORS[self._target_color_name]
        draw.rectangle([sx, sy, sx + swatch_w, sy + swatch_h], fill=target_rgb,
                        outline=(80, 80, 80))

        # Feedback from last step
        if self._last_result == "correct":
            draw.text((4, 4), "✓ correct", fill=_CORRECT_HL)
        elif self._last_result == "wrong":
            draw.text((4, 4), "✗ wrong", fill=_WRONG_HL)

        return img


# ── MCQButtonEnvironment ──────────────────────────────────────────────────────

@dataclass
class MCQQuestion:
    text: str                   # Question text
    choices: list[str]          # 4 answer labels
    correct_idx: int            # 0-based index of correct answer


def _make_dot_count_question(rng: np.random.Generator) -> MCQQuestion:
    """Count dots in image; choices are 1-4."""
    n = int(rng.integers(1, 5))  # 1 to 4
    choices = ["1 dot", "2 dots", "3 dots", "4 dots"]
    return MCQQuestion(
        text=f"How many dots are shown?",
        choices=choices,
        correct_idx=n - 1,
    )


def _make_color_id_question(rng: np.random.Generator) -> MCQQuestion:
    """Identify which circle is a specific color."""
    color_names = ["red", "green", "blue", "yellow"]
    answer_idx = int(rng.integers(4))
    target_color = color_names[answer_idx]
    choices = [f"Circle {chr(65+i)}" for i in range(4)]
    return MCQQuestion(
        text=f"Which circle is {target_color.upper()}?",
        choices=choices,
        correct_idx=answer_idx,
    )


class MCQButtonEnvironment(BaseEnvironment):
    """
    Multiple-choice question environment for MultiButtonAppendage (4 buttons).

    Each episode presents one question with 4 choices (A/B/C/D).
    The model presses the button corresponding to its answer.
    The button with the highest confidence is taken as the model's choice.

    Question types:
      - "dots"   : count dots on screen (1–4)
      - "colors" : identify which labeled circle is a specific color
      - "mixed"  : randomly alternates between types
    """

    N_BUTTONS = 4
    LABELS = ["A", "B", "C", "D"]

    def __init__(
        self,
        width: int = 280,
        height: int = 224,
        question_type: str = "mixed",   # "dots" | "colors" | "mixed"
        max_steps: int = 1,
    ):
        self.width = width
        self.height = height
        self.question_type = question_type
        self._max_steps = max_steps

        self._question: MCQQuestion | None = None
        self._dot_count: int = 0
        self._dot_positions: list[tuple[float, float]] = []
        self._circle_colors: list[str] = []
        self._step_count = 0
        self._last_result: str = ""
        self._rng = np.random.default_rng()

    @property
    def prompt(self) -> str:
        if self._question is None:
            return "Answer the multiple choice question. Press A, B, C, or D."
        return (
            f"{self._question.text} "
            f"Press the button for your answer: "
            f"A={self._question.choices[0]}, "
            f"B={self._question.choices[1]}, "
            f"C={self._question.choices[2]}, "
            f"D={self._question.choices[3]}."
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def reset(self, seed: int | None = None) -> Image.Image:
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_result = ""

        qtype = self.question_type
        if qtype == "mixed":
            qtype = "dots" if self._rng.random() < 0.5 else "colors"

        if qtype == "dots":
            self._question = _make_dot_count_question(self._rng)
            self._dot_count = self._question.correct_idx + 1
            self._dot_positions = self._random_dot_positions(self._dot_count)
        else:
            self._question = _make_color_id_question(self._rng)
            # Use the question's target color at the correct_idx
            color_names = ["red", "green", "blue", "yellow"]
            self._circle_colors = color_names.copy()

        return self._render()

    def step(self, action) -> EnvStepResult:
        # action is [batch, 4] logits/probabilities, or a MultiButtonState
        if hasattr(action, "confidences"):
            confidences = action.confidences
            chosen = max(range(len(confidences)), key=lambda i: confidences[i])
        elif isinstance(action, (list, np.ndarray)):
            arr = list(action)
            chosen = max(range(len(arr)), key=lambda i: float(arr[i]))
        else:
            chosen = int(action)

        correct = (chosen == self._question.correct_idx)
        self._step_count += 1

        if correct:
            reward = 15.0
            self._last_result = "correct"
        else:
            reward = -3.0
            self._last_result = "wrong"

        return EnvStepResult(
            observation=self._render(),
            reward=reward,
            done=True,
            info={
                "chosen": chosen,
                "correct_idx": self._question.correct_idx,
                "correct": correct,
                "success": correct,
                "question": self._question.text,
            },
        )

    def expert_action(self) -> list[float]:
        """One-hot for the correct answer button."""
        out = [0.0] * self.N_BUTTONS
        out[self._question.correct_idx] = 1.0
        return out

    def _random_dot_positions(self, n: int) -> list[tuple[float, float]]:
        margin = 30
        positions = []
        for _ in range(n):
            for _ in range(100):
                x = float(self._rng.uniform(margin, self.width - margin - 80))
                y = float(self._rng.uniform(margin + 40, self.height - 70))
                # Check minimum separation
                if all(math.hypot(x - px, y - py) > 28 for px, py in positions):
                    positions.append((x, y))
                    break
        return positions

    def _render(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), _BG)
        draw = ImageDraw.Draw(img)

        # Question box at top
        draw.rectangle([4, 4, self.width - 4, 36], fill=_PANEL_BG, outline=(180, 178, 175))
        q_text = self._question.text if self._question else "Loading..."
        draw.text((10, 12), q_text, fill=_TEXT_DARK)

        # Scene area
        scene_top = 44

        if self._question and "dots" in self._question.text.lower():
            # Render dots
            dot_r = 10
            for i, (dx, dy) in enumerate(self._dot_positions):
                draw.ellipse(
                    [dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r],
                    fill=(60, 60, 60),
                    outline=(30, 30, 30),
                    width=2,
                )

        elif self._question and "circle" in self._question.text.lower():
            # Render 4 colored circles labeled A/B/C/D
            color_names = ["red", "green", "blue", "yellow"]
            n = 4
            spacing = (self.width - 80) // n
            for i, cname in enumerate(color_names):
                cx = 40 + spacing * i + spacing // 2
                cy = scene_top + (self.height - scene_top - 70) // 2
                r = 22
                rgb = _NAMED_COLORS[cname]
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb,
                              outline=(max(0, rgb[0]-40), max(0, rgb[1]-40), max(0, rgb[2]-40)),
                              width=2)
                draw.text((cx - 4, cy + r + 4), self.LABELS[i], fill=_TEXT_DARK)

        # Answer buttons at bottom
        btn_w = (self.width - 20) // 4
        btn_h = 28
        btn_y = self.height - btn_h - 8

        for i, label in enumerate(self.LABELS):
            bx = 8 + i * (btn_w + 3)
            choice_text = f"{label}: {self._question.choices[i]}" if self._question else label

            # Highlight correct/chosen
            if self._last_result == "correct" and i == self._question.correct_idx:
                btn_color = _CORRECT_HL
            elif self._last_result == "wrong" and i == self._question.correct_idx:
                btn_color = _CORRECT_HL  # show correct answer
            else:
                btn_color = (210, 208, 205)

            draw.rectangle([bx, btn_y, bx + btn_w, btn_y + btn_h],
                            fill=btn_color, outline=(140, 138, 135), width=1)
            draw.text((bx + 4, btn_y + 7), choice_text[:14], fill=_TEXT_DARK)

        # Feedback
        if self._last_result == "correct":
            draw.text((4, 4), "✓", fill=_CORRECT_HL)
        elif self._last_result == "wrong":
            draw.text((4, 4), "✗", fill=_WRONG_HL)

        return img
