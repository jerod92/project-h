"""
VisionBridge — universal skip-connection from the VLM's vision encoder to any appendage.

Problem:
    By default, appendages receive only the LLM's last-token hidden state.
    This 1D vector has passed through the entire language decoder, compressing
    away fine-grained spatial information (object positions, distances, layout).
    This forces hundreds of BC steps to learn trivially simple spatial tasks.

Solution:
    VisionBridge wraps *any* BaseAppendage and injects mean-pooled vision
    encoder features as a skip connection.  The vision features are projected,
    fused with the LLM hidden state, and fed to the wrapped appendage — all
    without modifying the appendage's architecture.

Architecture::

    vision_features  → LayerNorm → Linear(vision_dim, hidden_dim) → GELU ─┐
                                                                           │
    llm_features     ─────────────────────────────────────────────────────→ + (add)
                                                                           │
                                                                  → appendage.forward()

Usage::

    from vla_hands import VisionBridge, JoystickAppendage, VLAGraft

    joystick = JoystickAppendage(hidden_dim=1152)
    bridged  = VisionBridge(joystick, vision_dim=1152)
    graft    = VLAGraft(vlm=vlm, appendage=bridged)
    # VLAGraft auto-detects needs_vision_features and hooks the encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..appendages.base import ActionSpec, BaseAppendage


class VisionBridge(BaseAppendage):
    """
    Wraps any BaseAppendage to add vision encoder skip connections.

    The bridge projects vision features to the appendage's hidden dimension,
    adds them to the LLM features (residual-style fusion), and delegates
    everything else to the wrapped appendage.  The wrapped appendage's
    forward(), action_loss(), decode(), and action_spec are all preserved
    unchanged.

    Args:
        appendage:  Any BaseAppendage instance (JoystickAppendage, DPadAppendage, etc.)
        vision_dim: Hidden size of the VLM's vision encoder patch embeddings.
                    Use ``VLAGraft.detect_vision_dim(vlm)`` to auto-detect.
        fusion:     How to combine vision + LLM features:
                    ``"add"`` (default) — additive residual (keeps dimension unchanged).
                    ``"gate"`` — learned sigmoid gate for adaptive blending.
    """

    #: VLAGraft checks this flag to register the vision encoder forward hook.
    needs_vision_features: bool = True

    def __init__(
        self,
        appendage: BaseAppendage,
        vision_dim: int,
        fusion: str = "add",
    ):
        super().__init__()
        self.wrapped = appendage
        self.vision_dim = vision_dim
        self.fusion_mode = fusion

        # Infer hidden_dim from the wrapped appendage
        self.hidden_dim = getattr(appendage, "hidden_dim", None)
        if self.hidden_dim is None:
            raise ValueError(
                f"{type(appendage).__name__} does not expose a hidden_dim attribute. "
                "VisionBridge needs to know the appendage's input dimension."
            )

        # Vision feature projection: vision_dim → hidden_dim
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, self.hidden_dim),
            nn.GELU(),
        )

        # Optional gating mechanism
        if fusion == "gate":
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        self._init_vision_weights()

    def _init_vision_weights(self):
        """Small-magnitude init so bridge starts as near-identity (LLM features dominate)."""
        for m in self.vision_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        if self.gate is not None:
            for m in self.gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    # Bias to 0.5 → equal initial blend
                    nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------ #
    #  BaseAppendage interface — delegated to wrapped appendage            #
    # ------------------------------------------------------------------ #

    @property
    def action_spec(self) -> ActionSpec:
        return self.wrapped.action_spec

    def forward(
        self,
        features: torch.Tensor,                     # [batch, hidden_dim]
        vision_features: torch.Tensor | None = None, # [N, vision_dim]  (N >= batch)
    ) -> torch.Tensor:
        """
        Fuse vision + LLM features, then delegate to the wrapped appendage.

        When vision_features is None (e.g. no vision hook registered), this
        behaves identically to calling the wrapped appendage directly.

        Note:
            Some VLMs (SmolVLM / Idefics3) split each input image into multiple
            sub-images before processing through the vision encoder.  This means
            vision_features may have a larger batch dimension than ``features``.
            When this happens, the sub-image features are grouped and mean-pooled
            back to the original batch size automatically.
        """
        if vision_features is not None:
            batch_size = features.shape[0]
            n_vision = vision_features.shape[0]

            if n_vision != batch_size and n_vision > batch_size and n_vision % batch_size == 0:
                # VLM split each image into N sub-images; pool them back
                n_sub = n_vision // batch_size
                vision_features = vision_features.reshape(batch_size, n_sub, -1).mean(dim=1)
            elif n_vision != batch_size:
                # Fallback: just mean-pool everything into a single vector and broadcast
                vision_features = vision_features.mean(dim=0, keepdim=True).expand(batch_size, -1)

            v = self.vision_proj(vision_features)  # [batch, hidden_dim]

            if self.gate is not None:
                # Learned gating: α ∈ [0, 1] per dimension
                alpha = self.gate(torch.cat([features, v], dim=-1))
                features = features * (1 - alpha) + v * alpha
            else:
                # Simple additive residual
                features = features + v

        return self.wrapped(features)

    def action_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.wrapped.action_loss(predicted, target, weights)

    def decode(self, action_tensor: torch.Tensor):
        return self.wrapped.decode(action_tensor)

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"VisionBridge(\n"
            f"  wrapped={self.wrapped!r},\n"
            f"  vision_dim={self.vision_dim},\n"
            f"  fusion={self.fusion_mode!r},\n"
            f"  bridge_params={sum(p.numel() for p in self.vision_proj.parameters()):,}\n"
            f")"
        )
