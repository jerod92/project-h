"""
D-pad appendage — discrete 5-way directional control.

Buttons: STAY, UP, DOWN, LEFT, RIGHT

Suitable for:
  - Grid-world navigation
  - Menu / UI navigation
  - Turn-based game control
  - Any discrete directional task

Architecture:
    hidden_state → LayerNorm → Linear(hidden, mid) → GELU
                → Linear(mid, 64)  → GELU
                → Linear(64, 5)    →  raw logits  (5 classes)

No final softmax — callers use argmax for greedy inference or
sample via Categorical for exploration. CrossEntropy loss for training.
"""

from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActionSpec, BaseAppendage


class DPadButton(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    def __repr__(self) -> str:
        return f"DPadButton.{self.name}"


# (dx, dy) in grid coordinates.
# dx: column delta (+1 = right), dy: row delta (+1 = down, -1 = up)
DPAD_DELTA: dict[DPadButton, tuple[int, int]] = {
    DPadButton.STAY:  (0,  0),
    DPadButton.UP:    (0, -1),
    DPadButton.DOWN:  (0,  1),
    DPadButton.LEFT:  (-1, 0),
    DPadButton.RIGHT: (1,  0),
}

BUTTON_NAMES = {b: b.name for b in DPadButton}


class DPadAppendage(BaseAppendage):
    """
    Discrete D-pad action head with 5 buttons: STAY, UP, DOWN, LEFT, RIGHT.

    Outputs raw logits [batch, 5]. Use argmax() for greedy inference or
    sample() for stochastic exploration during RL.
    """

    N_ACTIONS = len(DPadButton)

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
            nn.Linear(64, self.N_ACTIONS),
        )

        self._action_spec = ActionSpec(
            name="dpad",
            shape=(self.N_ACTIONS,),
            dtype="discrete",
            n_discrete=self.N_ACTIONS,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small gain on output layer only — start with near-uniform logits
        # without crushing gradient flow through hidden layers.
        output_linear = self.net[-1]  # Linear(64, N_ACTIONS), no activation after
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
            logits: [batch, N_ACTIONS=5]  (raw, no softmax)
        """
        return self.net(hidden_state)

    def action_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Cross-entropy loss.

        Args:
            predicted: Logits [batch, N_ACTIONS].
            target: Integer class indices [batch] (values in 0..N_ACTIONS-1).
            weights: Optional per-sample importance weights [batch].
        """
        loss = F.cross_entropy(predicted, target.long(), reduction="none")  # [batch]
        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        return loss

    def argmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy: return the highest-scoring action index."""
        return torch.argmax(logits, dim=-1)

    def sample(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample an action from the softmax distribution."""
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def decode(self, action_tensor: torch.Tensor) -> DPadButton:
        t = action_tensor.detach().cpu()
        if t.dim() > 0 and t.shape[-1] == self.N_ACTIONS:
            idx = int(t.argmax(-1).item())
        else:
            idx = int(t.item())
        return DPadButton(idx)
