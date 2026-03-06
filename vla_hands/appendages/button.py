"""
Button appendages — binary press actions.

ButtonAppendage:
    Single button with sigmoid output in [0, 1].
    Threshold at 0.5 → pressed / not pressed.
    Loss: binary cross-entropy.
    Suitable for: yes/no decisions, sentence-end markers, collect/interact actions.

MultiButtonAppendage:
    N independent binary buttons, each with sigmoid output.
    Each button is predicted independently (not mutually exclusive like softmax).
    Loss: mean BCE across all N buttons.
    Suitable for: multiple-choice answering, multi-action panels, chord keys.

Design note on independence:
    Multi-button uses independent sigmoids (not softmax) because real button panels
    allow simultaneous presses. Use DPadAppendage if you need exactly-one semantics.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActionSpec, BaseAppendage


# ── ButtonAction helper ───────────────────────────────────────────────────────

@dataclass
class ButtonState:
    """Human-readable state for a single button."""
    pressed: bool
    confidence: float  # raw sigmoid value

    def __repr__(self) -> str:
        sym = "■" if self.pressed else "□"
        return f"Button[{sym} {self.confidence:.3f}]"


@dataclass
class MultiButtonState:
    """Human-readable state for a multi-button panel."""
    pressed: list[bool]
    confidences: list[float]

    def __repr__(self) -> str:
        buttons = "".join("■" if p else "□" for p in self.pressed)
        return f"MultiButton[{buttons}]"

    def any_pressed(self) -> bool:
        return any(self.pressed)

    def pressed_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.pressed) if p]


# ── ButtonAppendage ───────────────────────────────────────────────────────────

class ButtonAppendage(BaseAppendage):
    """
    Single binary button action head.

    Output: scalar sigmoid ∈ [0, 1].
    Decision: pressed ↔ value > threshold (default 0.5).

    Architecture:
        hidden → LayerNorm → Linear(H, 128) → GELU → Linear(128, 32) → GELU
               → Linear(32, 1) → Sigmoid
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = 128,
        press_threshold: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.press_threshold = press_threshold

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._action_spec = ActionSpec(
            name="button",
            shape=(1,),
            dtype="continuous",
            low=0.0,
            high=1.0,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small gain on output layer only — start predictions at sigmoid(0)=0.5
        # without crushing gradient flow through hidden layers.
        output_linear = self.net[-2]  # Linear(32, 1), before Sigmoid
        nn.init.orthogonal_(output_linear.weight, gain=0.01)
        nn.init.zeros_(output_linear.bias)

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_dim]
        Returns:
            confidence: [batch, 1] ∈ [0, 1]
        """
        return self.net(hidden_state)

    def action_loss(
        self,
        predicted: torch.Tensor,           # [batch, 1]
        target: torch.Tensor,              # [batch, 1] or [batch] float {0, 1}
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Binary cross-entropy loss."""
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        target = target.float()
        loss = F.binary_cross_entropy(predicted, target, reduction="none")  # [batch, 1]
        loss = loss.squeeze(-1)  # [batch]
        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        return loss

    def is_pressed(self, output: torch.Tensor) -> torch.Tensor:
        """Returns bool tensor: pressed ↔ output > threshold."""
        return output.squeeze(-1) > self.press_threshold

    def decode(self, action_tensor: torch.Tensor) -> ButtonState:
        t = action_tensor.detach().cpu().float()
        if t.dim() > 1:
            t = t[0]
        val = float(t.squeeze())
        return ButtonState(pressed=val > self.press_threshold, confidence=val)


# ── MultiButtonAppendage ──────────────────────────────────────────────────────

class MultiButtonAppendage(BaseAppendage):
    """
    N independent binary buttons, each predicted with its own sigmoid.

    Unlike DPadAppendage (softmax = exactly one), MultiButton allows
    zero, one, or many buttons pressed simultaneously.

    Architecture:
        hidden → LayerNorm → Linear(H, 256) → GELU → Linear(256, 64) → GELU
               → Linear(64, N) → Sigmoid

    Args:
        n_buttons: Number of independent buttons.
        labels: Optional list of human-readable button names (e.g. ["A","B","C","D"]).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_buttons: int,
        labels: list[str] | None = None,
        intermediate_dim: int = 256,
        press_threshold: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_buttons = n_buttons
        self.labels = labels or [str(i) for i in range(n_buttons)]
        self.press_threshold = press_threshold

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_buttons),
            nn.Sigmoid(),
        )
        self._action_spec = ActionSpec(
            name=f"multi_button_{n_buttons}",
            shape=(n_buttons,),
            dtype="continuous",
            low=0.0,
            high=1.0,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small gain on output layer only — start predictions at sigmoid(0)=0.5
        # without crushing gradient flow through hidden layers.
        output_linear = self.net[-2]  # Linear(64, n_buttons), before Sigmoid
        nn.init.orthogonal_(output_linear.weight, gain=0.01)
        nn.init.zeros_(output_linear.bias)

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_dim]
        Returns:
            confidences: [batch, n_buttons] ∈ [0, 1] each
        """
        return self.net(hidden_state)

    def action_loss(
        self,
        predicted: torch.Tensor,          # [batch, n_buttons]
        target: torch.Tensor,             # [batch, n_buttons] float {0, 1}
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Mean BCE across all buttons.

        For single-correct-answer tasks (MCQ), pass target as one-hot float.
        For multi-press tasks, pass target as multi-hot float.
        """
        target = target.float()
        loss = F.binary_cross_entropy(predicted, target, reduction="none")  # [batch, N]
        loss = loss.mean(dim=-1)  # [batch] — average over buttons
        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        return loss

    def argmax_button(self, output: torch.Tensor) -> torch.Tensor:
        """Return index of highest-confidence button (for MCQ-style tasks)."""
        return output.argmax(dim=-1)

    def pressed_buttons(self, output: torch.Tensor) -> list[list[int]]:
        """Return list of pressed button indices per batch item."""
        pressed = output > self.press_threshold
        result = []
        for row in pressed:
            result.append([i for i, p in enumerate(row) if p])
        return result

    def decode(self, action_tensor: torch.Tensor) -> MultiButtonState:
        t = action_tensor.detach().cpu().float()
        if t.dim() > 1:
            t = t[0]
        vals = t.tolist()
        return MultiButtonState(
            pressed=[v > self.press_threshold for v in vals],
            confidences=vals,
        )
