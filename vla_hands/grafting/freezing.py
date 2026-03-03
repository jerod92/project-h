"""
Freezing curriculum — gradual unfreezing of VLM weights during training.

Rationale (answering the design question):

  The grafting process should be a *staged* discrete curriculum, not a continuous
  thaw. Here's why:

  1. Appendage-first (Stage 0):
     Start with ALL VLM weights frozen. The action head MLP must learn to read
     the frozen VLM's representations. This is fast, stable, and proves whether
     the VLM's existing features are sufficient for the action task.

  2. Downstream unfreeze (Stages 1+):
     Slowly unfreeze transformer layers from the END of the model backwards
     (closest to the action head → furthest from the embedding). The final layers
     can then refine their representations to be more action-useful.
     Early layers (embeddings, early transformer blocks) are left frozen to
     preserve the VLM's foundational language and visual understanding.

  3. Full fine-tune (optional Stage N):
     Only attempt if downstream stages plateau and you have enough data.
     Risks catastrophic forgetting of language ability.

  The language generation path is NEVER removed or altered. We add a branch;
  the trunk stays intact. So we never "need" to thaw the layers above the graft —
  language remains fully functional throughout.

  Key heuristic: use LR ≈ 1/10 of appendage LR for unfrozen VLM layers, and
  ≈ 1/100 for very early layers if doing full fine-tune.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn


@dataclass
class FreezingStage:
    """One stage in the freezing curriculum."""

    name: str
    description: str

    # Number of transformer layers to unfreeze from the output end.
    # 0 = keep all VLM weights frozen (train action head only).
    # Large values (e.g. 999) effectively unfreeze all layers.
    n_layers_unfrozen: int

    # Global training step at which this stage activates.
    start_step: int = 0

    # Whether to also unfreeze the vision encoder (e.g. CLIP ViT).
    unfreeze_vision: bool = False

    # Whether to also unfreeze the LM head (output projection).
    unfreeze_lm_head: bool = False


# ── Preset stages ────────────────────────────────────────────────────────────

STAGE_APPENDAGE_ONLY = FreezingStage(
    name="appendage_only",
    description="All VLM weights frozen — train action head only. Fast convergence.",
    n_layers_unfrozen=0,
    start_step=0,
)

STAGE_LAST_2 = FreezingStage(
    name="last_2_layers",
    description="Unfreeze final 2 transformer layers. Representation adaptation.",
    n_layers_unfrozen=2,
    start_step=500,
)

STAGE_LAST_6 = FreezingStage(
    name="last_6_layers",
    description="Unfreeze final 6 transformer layers. Deeper adaptation.",
    n_layers_unfrozen=6,
    start_step=1500,
)

STAGE_FULL = FreezingStage(
    name="full_finetune",
    description="Full model fine-tuning with very small LR. Use with caution.",
    n_layers_unfrozen=9999,
    unfreeze_vision=True,
    unfreeze_lm_head=True,
    start_step=4000,
)

# Default curriculum stops at STAGE_LAST_6 — does not do full fine-tuning.
DEFAULT_CURRICULUM: list[FreezingStage] = [
    STAGE_APPENDAGE_ONLY,
    STAGE_LAST_2,
    STAGE_LAST_6,
]

# Minimal curriculum for quick experiments
QUICK_CURRICULUM: list[FreezingStage] = [
    STAGE_APPENDAGE_ONLY,
]


# ── FreezingCurriculum ────────────────────────────────────────────────────────

# Common attribute paths for finding transformer layer lists in HF models
_LAYER_PATHS = [
    "model.layers",
    "language_model.model.layers",
    "model.language_model.model.layers",
    "model.decoder.layers",
    "transformer.h",
    "model.transformer.h",
    "gpt_neox.layers",
    "model.model.layers",
]

# Common attribute paths for the vision encoder
_VISION_PATHS = [
    "vision_model",
    "vision_tower",
    "visual_encoder",
    "vision_encoder",
    "model.vision_model",
]

# Common attribute paths for the LM head
_LM_HEAD_PATHS = ["lm_head", "embed_out", "output_projection", "output"]


class FreezingCurriculum:
    """
    Manages staged, step-based unfreezing of VLM parameters.

    Usage::

        curriculum = FreezingCurriculum(vlm=model)
        optimizer = ...

        for step in range(total_steps):
            changed = curriculum.step(step)  # updates requires_grad
            if changed:
                optimizer = rebuild_optimizer(graft)  # rebuild with new params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        vlm: nn.Module,
        stages: list[FreezingStage] = DEFAULT_CURRICULUM,
        on_stage_change: Callable[[FreezingStage], None] | None = None,
    ):
        self.vlm = vlm
        self.stages = sorted(stages, key=lambda s: s.start_step)
        self.on_stage_change = on_stage_change
        self._current_stage_idx = -1

        # Begin with all VLM weights frozen
        self._freeze_all()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def step(self, global_step: int) -> bool:
        """
        Advance the curriculum based on the current training step.

        Should be called once per optimizer step, before loss.backward().

        Returns:
            True if the stage changed (caller should rebuild the optimizer
            to include newly unfrozen parameters).
        """
        new_idx = self._current_stage_idx
        for i, stage in enumerate(self.stages):
            if global_step >= stage.start_step:
                new_idx = i
            else:
                break

        if new_idx != self._current_stage_idx:
            self._current_stage_idx = new_idx
            stage = self.stages[new_idx]
            self._apply_stage(stage)
            if self.on_stage_change:
                self.on_stage_change(stage)
            return True
        return False

    def apply_stage(self, stage: FreezingStage):
        """Manually apply a specific stage (bypasses step counter)."""
        idx = self.stages.index(stage)
        self._current_stage_idx = idx
        self._apply_stage(stage)

    @property
    def current_stage(self) -> FreezingStage | None:
        if self._current_stage_idx < 0:
            return None
        return self.stages[self._current_stage_idx]

    def frozen_param_count(self) -> tuple[int, int]:
        """Returns (n_trainable, n_total) for the VLM."""
        n_total = sum(p.numel() for p in self.vlm.parameters())
        n_trainable = sum(p.numel() for p in self.vlm.parameters() if p.requires_grad)
        return n_trainable, n_total

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _freeze_all(self):
        for p in self.vlm.parameters():
            p.requires_grad_(False)

    def _apply_stage(self, stage: FreezingStage):
        self._freeze_all()

        layers = self._find_transformer_layers()
        n = min(stage.n_layers_unfrozen, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)

        if stage.unfreeze_vision:
            for path in _VISION_PATHS:
                module = self._get_nested(self.vlm, path)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad_(True)

        if stage.unfreeze_lm_head:
            for path in _LM_HEAD_PATHS:
                module = self._get_nested(self.vlm, path)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad_(True)

        n_trainable, n_total = self.frozen_param_count()
        pct = 100.0 * n_trainable / max(n_total, 1)
        print(
            f"[FreezingCurriculum] → {stage.name} | "
            f"VLM trainable: {n_trainable:,} / {n_total:,}  ({pct:.1f}%)"
        )

    def _find_transformer_layers(self) -> list[nn.Module]:
        for path in _LAYER_PATHS:
            obj = self._get_nested(self.vlm, path)
            if obj is not None and isinstance(obj, (nn.ModuleList, list)):
                return list(obj)
        return []

    @staticmethod
    def _get_nested(obj: nn.Module, dotted_path: str) -> nn.Module | None:
        try:
            for attr in dotted_path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            return None
