"""
Base class and specifications for all VLA action appendages.

An appendage is a learned module grafted onto a VLM's hidden state to produce
structured actions in parallel with normal token generation.

Design goals:
  - Minimal interface: any appendage just needs forward() + action_loss()
  - Self-describing via ActionSpec (shape, dtype, bounds)
  - Composable: multiple appendages can be attached to a single graft
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ActionSpec:
    """Describes the shape and semantics of an action space."""

    name: str
    shape: tuple[int, ...]
    # "continuous" → real-valued tensor; "discrete" → integer class index
    dtype: str
    # Bounds for continuous spaces
    low: float | None = None
    high: float | None = None
    # Number of classes for discrete spaces
    n_discrete: int | None = None

    def __post_init__(self):
        if self.dtype == "continuous" and (self.low is None or self.high is None):
            raise ValueError("Continuous ActionSpec requires low and high bounds.")
        if self.dtype == "discrete" and self.n_discrete is None:
            raise ValueError("Discrete ActionSpec requires n_discrete.")


class BaseAppendage(nn.Module, ABC):
    """
    Abstract base class for all VLA action appendages.

    Subclasses implement a specific action space (joystick, d-pad, button, pointer…).
    They receive a feature vector extracted from the VLM's last hidden layer and output
    an action tensor.

    The appendage is always a pure nn.Module — it never touches the VLM backbone itself.
    The VLAGraft handles routing hidden states to appendages.
    """

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        """Describes the output action space of this appendage."""
        ...

    @abstractmethod
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Map VLM hidden-state features to an action tensor.

        Args:
            hidden_state: Float tensor of shape [batch, hidden_dim].
                          This is the feature vector extracted from the VLM's
                          final transformer layer (typically the last-token position).

        Returns:
            Tensor matching self.action_spec.shape (batch dimension prepended).
            Continuous: values in [low, high].
            Discrete: raw logits over n_discrete classes.
        """
        ...

    @abstractmethod
    def action_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute training loss between predicted and target actions.

        Args:
            predicted: Output of forward().
            target: Ground-truth action tensor.
            weights: Optional per-sample importance weights [batch].

        Returns:
            Scalar loss tensor.
        """
        ...

    def decode(self, action_tensor: torch.Tensor) -> Any:
        """Convert a raw action tensor to a human-readable Python object."""
        return action_tensor.detach().cpu().tolist()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
