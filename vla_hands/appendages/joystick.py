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
                → Linear(64, 2)    →  pre-squash logits (unbounded)

The network deliberately omits the final Tanh activation. Tanh squashing is
applied *externally* by the trainer (RL sampling) and action_loss (BC).

Rationale: if Tanh is baked into the network, BC training drives pre-tanh values
to atanh(target) ≈ 1.5–2.5 for unit-magnitude expert actions. In the subsequent
RL phase, exploration noise (std σ) is added in pre-tanh space, but the effective
std in action space shrinks to σ·(1 − tanh²(u_mean)). At u_mean=2 this is only
~8% of σ — the policy is essentially frozen after BC. Keeping the network in
unbounded (logit) space ensures σ is always applied in a consistent regime.
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
            # No Tanh here — see module docstring for why.
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
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small gain on output layer — start logits near zero so the squashed
        # action is near the origin, giving maximum exploration headroom at RL init.
        output_linear = self.net[-1]  # Linear(64, 2), now the final layer
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
            logits: [batch, 2] — pre-squash, unbounded. Apply torch.tanh() to get
                    the action in [-1, 1]². The trainer and action_loss do this
                    automatically; callers that want a bounded action should call
                    torch.tanh(appendage(features)).
        """
        return self.net(hidden_state)

    def action_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Huber loss between squashed prediction and target joystick positions.

        predicted is the raw pre-squash logit from forward(); tanh is applied here
        so the loss is always computed in the bounded action space [-1, 1]².
        Huber is more robust than MSE to occasional large expert-policy deviations.
        """
        squashed = torch.tanh(predicted)
        loss = F.huber_loss(squashed, target, delta=0.5, reduction="none")  # [batch, 2]
        if weights is not None:
            loss = (loss * weights.unsqueeze(-1)).mean()
        else:
            loss = loss.mean()
        return loss

    def decode(self, action_tensor: torch.Tensor) -> JoystickAction:
        t = action_tensor.detach().cpu().float()
        if t.dim() > 1:
            t = t[0]
        # action_tensor may be either pre-squash logits (from forward()) or
        # already-squashed values (from _select_action). Apply tanh only when
        # values are outside [-1, 1], i.e. clearly in logit space.
        if t.abs().max() > 1.0:
            t = torch.tanh(t)
        return JoystickAction(x=float(t[0]), y=float(t[1]))
