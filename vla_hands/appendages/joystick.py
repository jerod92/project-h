"""
Joystick appendage — continuous 2-axis control in [-1, 1]².

Suitable for:
  - Continuous navigation (move agent to target)
  - Camera pan/tilt control
  - Robot arm end-effector velocity
  - Pointer / cursor control

Architecture:
    hidden_state → LayerNorm → Linear(hidden, mid) → GELU
                → Linear(mid, 64)  → GELU
                → Linear(64, 2)    → Tanh  →  (x, y) ∈ [-1, 1]²

The Tanh naturally constrains the output range. Orthogonal weight init with
small gain (0.01) ensures the head starts near zero, letting the VLM's
pre-existing representations guide early training without catastrophic gradients.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActionSpec, BaseAppendage


class JoystickAction(NamedTuple):
    """Human-readable joystick action."""

    x: float  # [-1 = full left,  +1 = full right]
    y: float  # [-1 = full down,  +1 = full up]

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([self.x, self.y], dtype=torch.float32)

    def __repr__(self) -> str:
        return f"JoystickAction(x={self.x:+.3f}, y={self.y:+.3f})"


class JoystickAppendage(BaseAppendage):
    """
    Continuous 2D joystick action head.

    Outputs (x, y) ∈ [-1, 1]² in parallel with the VLM's token stream.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int = 256):
        """
        Args:
            hidden_dim: Dimensionality of the VLM's hidden states.
            intermediate_dim: Width of the first MLP layer.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

        self._action_spec = ActionSpec(
            name="joystick",
            shape=(2,),
            dtype="continuous",
            low=-1.0,
            high=1.0,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_dim]
        Returns:
            action: [batch, 2] with values in [-1, 1]
        """
        return self.net(hidden_state)

    def action_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Huber loss between predicted and target joystick positions.
        Huber is more robust than MSE to occasional large expert-policy deviations.
        """
        loss = F.huber_loss(predicted, target, delta=0.5, reduction="none")  # [batch, 2]
        if weights is not None:
            loss = (loss * weights.unsqueeze(-1)).mean()
        else:
            loss = loss.mean()
        return loss

    def decode(self, action_tensor: torch.Tensor) -> JoystickAction:
        t = action_tensor.detach().cpu().float()
        if t.dim() > 1:
            t = t[0]
        return JoystickAction(x=float(t[0]), y=float(t[1]))
