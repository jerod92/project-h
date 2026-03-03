"""
Touchscreen appendage — outputs absolute (x, y) screen coordinates.

Unlike the joystick (which outputs a *direction* vector in [-1, 1]²), the
touchscreen outputs a *position* on the screen, normalised to [0, 1]² where
(0, 0) is the top-left corner and (1, 1) is the bottom-right corner.

This is the right primitive for pointing, clicking, and tapping tasks where
the model needs to identify *where* on the image to interact, not *which
direction* to move.

Architecture
------------
The head combines two information sources:

  1. LLM hidden state  — the final-layer contextual representation.  This
     encodes the semantic understanding of the scene: which object matches the
     task description and therefore *should* be tapped.

  2. Vision skip connection  — spatial features from the VLM's vision encoder
     (optional but recommended).  The vision encoder patch embeddings carry
     rich spatial information about where objects sit in the image.  By skip-
     connecting these features directly into the action head we bypass the
     information bottleneck of the LLM decoder, letting the head "look at" the
     raw visual layout.

When vision_dim is None (or no hook is registered), the head operates on LLM
features alone — still useful but less spatially precise.

Usage
-----
    # Without vision skip:
    head = TouchscreenAppendage(hidden_dim=1152)

    # With vision skip (requires VLAGraft to register a vision hook):
    head = TouchscreenAppendage(hidden_dim=1152, vision_dim=1152)

    # VLAGraft auto-detects the vision encoder; set vision_dim by inspecting:
    vision_dim = VLAGraft.detect_vision_dim(vlm)   # helper
    head = TouchscreenAppendage(hidden_dim, vision_dim=vision_dim)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActionSpec, BaseAppendage


class TouchPoint(NamedTuple):
    """A screen coordinate in [0, 1]²."""
    x: float   # horizontal (0 = left,  1 = right)
    y: float   # vertical   (0 = top,   1 = bottom)


class TouchscreenAppendage(BaseAppendage):
    """
    Action head that predicts an absolute screen tap coordinate.

    Output: (x, y) ∈ [0, 1]² via Sigmoid activation.
    Loss:   Huber loss (smooth L1, delta=0.1) between predicted and target coordinates.

    Optionally takes `vision_features` from the VLM's vision encoder as a skip
    connection.  When provided, vision features are projected to the same
    intermediate dimension and concatenated with the LLM features before the
    final prediction layers.
    """

    #: VLAGraft checks this flag to decide whether to register a vision hook.
    needs_vision_features: bool = True

    def __init__(
        self,
        hidden_dim: int,
        vision_dim: int | None = None,
        intermediate_dim: int = 256,
    ):
        """
        Args:
            hidden_dim: Size of the LLM's final hidden state (e.g. 1152 for
                        SmolVLM-256M, 2048 for LLaMA-3.2-1B-based models).
            vision_dim: Hidden size of the vision encoder patch embeddings.
                        When provided, a skip-connection MLP is built.
                        Pass None (default) to use LLM features only.
            intermediate_dim: Width of the shared projection MLP.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vision_dim = vision_dim
        self.intermediate_dim = intermediate_dim

        # LLM feature branch
        self.llm_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
        )

        # Vision skip branch (only built when vision_dim is specified)
        if vision_dim is not None:
            self.vision_proj: nn.Module = nn.Sequential(
                nn.LayerNorm(vision_dim),
                nn.Linear(vision_dim, intermediate_dim),
                nn.GELU(),
            )
            combined_dim = intermediate_dim * 2
        else:
            self.vision_proj = None  # type: ignore[assignment]
            combined_dim = intermediate_dim

        # Final prediction head
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, intermediate_dim // 2),
            nn.GELU(),
            nn.Linear(intermediate_dim // 2, 2),  # → (x, y) pre-sigmoid
        )

    # ------------------------------------------------------------------ #
    #  BaseAppendage interface                                             #
    # ------------------------------------------------------------------ #

    @property
    def action_spec(self) -> ActionSpec:
        return ActionSpec(
            name="touchscreen",
            shape=(2,),
            dtype="continuous",
            low=0.0,
            high=1.0,
        )

    def forward(
        self,
        features: torch.Tensor,                     # [batch, hidden_dim]
        vision_features: torch.Tensor | None = None, # [batch, vision_dim]
    ) -> torch.Tensor:                               # [batch, 2]
        """
        Predict a tap coordinate from LLM (+ optional vision) features.

        Args:
            features:        LLM hidden state, shape [batch, hidden_dim].
            vision_features: Mean-pooled vision encoder output, shape
                             [batch, vision_dim].  Ignored when vision_proj
                             is None or when None is passed.

        Returns:
            Tensor of shape [batch, 2] with values in [0, 1].
        """
        x = self.llm_proj(features)   # [batch, intermediate_dim]

        if self.vision_proj is not None and vision_features is not None:
            v = self.vision_proj(vision_features)   # [batch, intermediate_dim]
            x = torch.cat([x, v], dim=-1)           # [batch, intermediate_dim * 2]

        logits = self.predictor(x)          # [batch, 2]
        return torch.sigmoid(logits)        # → [0, 1]²

    def action_loss(
        self,
        pred: torch.Tensor,    # [batch, 2]  predicted coordinates
        target: torch.Tensor,  # [batch, 2]  expert coordinates
    ) -> torch.Tensor:
        """Huber loss (delta=0.1) — robust to occasional large prediction errors."""
        return F.huber_loss(pred, target, delta=0.1)

    def decode(self, action_tensor: torch.Tensor) -> TouchPoint:
        """Convert a [1, 2] or [2] tensor to a TouchPoint namedtuple."""
        t = action_tensor.detach().squeeze(0)
        return TouchPoint(x=float(t[0]), y=float(t[1]))

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        skip = f", vision_dim={self.vision_dim}" if self.vision_dim else ""
        return (
            f"TouchscreenAppendage("
            f"hidden_dim={self.hidden_dim}"
            f"{skip}, "
            f"intermediate_dim={self.intermediate_dim}, "
            f"params={self.num_parameters():,})"
        )
