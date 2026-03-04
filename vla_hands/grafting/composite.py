"""
CompositeGraft — multiple action appendages sharing one VLM backbone.

Instead of running the expensive VLM forward pass separately for each
appendage, CompositeGraft runs it once and fans out the hidden state to N
independent action heads.  This is essential for multi-appendage environments
(e.g. navigate-and-press) where both actions must be computed from the same
observation.

Architecture::

    image + prompt
         ↓
    [VLM backbone]  (one forward pass, output_hidden_states=True)
         ↓
    last-token hidden state
         ├── [JoystickAppendage] → (x, y) ∈ [-1,1]²
         ├── [ButtonAppendage]   → press ∈ [0,1]
         └── [TouchscreenAppendage] → (x, y) ∈ [0,1]²

Usage::

    from vla_hands import VLAGraft, GraftConfig
    from vla_hands.grafting.composite import CompositeGraft
    from vla_hands import JoystickAppendage, ButtonAppendage

    hidden_dim = 1152
    graft = CompositeGraft(
        vlm=vlm,
        appendages={
            "joystick": JoystickAppendage(hidden_dim),
            "button":   ButtonAppendage(hidden_dim),
        },
    )

    # forward() returns a dict of tensors keyed by appendage name
    out = graft(**inputs)
    # → {"joystick": Tensor[1,2], "button": Tensor[1,1], "lm_logits": Tensor[...]}

    graft.save("my_ckpt/")
    graft.load_appendages("my_ckpt/")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .graft import GraftConfig, _VisionHook
from ..appendages.base import BaseAppendage


class CompositeGraft(nn.Module):
    """
    Single VLM backbone driving multiple action appendages simultaneously.

    The VLM runs exactly once per forward pass.  Each appendage receives the
    same feature vector (last non-padding hidden state) and produces its own
    action output independently.

    This is more efficient than running N separate VLAGraft instances (one per
    appendage) when the environment requires joint action predictions.
    """

    def __init__(
        self,
        vlm: nn.Module,
        appendages: dict[str, BaseAppendage],
        config: GraftConfig | None = None,
    ):
        """
        Args:
            vlm:        HuggingFace VLM (any model compatible with VLAGraft).
            appendages: Dict mapping action name → appendage instance.
                        E.g. {"joystick": JoystickAppendage(hd), "button": ButtonAppendage(hd)}
            config:     GraftConfig; hidden_dim auto-detected if None.
        """
        super().__init__()
        self.vlm = vlm
        self.appendages = nn.ModuleDict(appendages)
        self.config = config or GraftConfig()

        if self.config.hidden_dim is None:
            self.config.hidden_dim = self._detect_hidden_dim()

        # Register vision hooks for appendages that need them
        self._vision_hooks: dict[str, _VisionHook] = {}
        vision_encoder = None
        for name, app in appendages.items():
            if getattr(app, "needs_vision_features", False):
                if vision_encoder is None:
                    vision_encoder = self._find_vision_encoder()
                if vision_encoder is not None:
                    # Share one hook — all vision-needing appendages see the same features
                    if "_shared" not in self._vision_hooks:
                        self._vision_hooks["_shared"] = _VisionHook(vision_encoder)
                else:
                    import warnings
                    warnings.warn(
                        f"CompositeGraft: appendage '{name}' needs vision features but "
                        "vision encoder not found. Running without skip connection.",
                        stacklevel=2,
                    )

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        One VLM forward pass → dict of action tensors from each appendage.

        Returns:
            dict with one key per appendage (e.g. "joystick", "button") plus:
              "lm_logits"       — language model logits
              "action_features" — shared hidden state used as input to all heads
              "lm_loss"         — only present when labels is not None
        """
        vlm_out = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden_states = vlm_out.hidden_states[-1]
        action_features = self._extract_features(hidden_states, attention_mask)

        # Shared vision skip features (if any appendage requested them)
        vision_features = None
        if "_shared" in self._vision_hooks:
            hook = self._vision_hooks["_shared"]
            if hook.features is not None:
                vision_features = hook.features

        result: dict[str, torch.Tensor] = {
            "lm_logits": vlm_out.logits,
            "action_features": action_features,
        }
        if getattr(vlm_out, "loss", None) is not None:
            result["lm_loss"] = vlm_out.loss

        for name, app in self.appendages.items():
            needs_vis = getattr(app, "needs_vision_features", False)
            if needs_vis and vision_features is not None:
                result[name] = app(action_features, vision_features=vision_features)
            else:
                result[name] = app(action_features)

        return result

    @torch.no_grad()
    def predict_actions(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Inference wrapper: returns only the action tensors (no gradients)."""
        out = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return {k: v for k, v in out.items()
                if k not in ("lm_logits", "action_features", "lm_loss")}

    # ------------------------------------------------------------------ #
    #  Training helpers                                                   #
    # ------------------------------------------------------------------ #

    def parameter_groups(
        self,
        appendage_lr: float = 1e-4,
        vlm_lr: float = 1e-6,
    ) -> list[dict]:
        """AdamW parameter groups with separate LRs for appendages vs. VLM."""
        groups = [{"params": list(self.appendages.parameters()), "lr": appendage_lr}]
        vlm_params = [p for p in self.vlm.parameters() if p.requires_grad]
        if vlm_params:
            groups.append({"params": vlm_params, "lr": vlm_lr})
        return groups

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                      #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """
        Save all appendage weights to ``path/appendage_{name}.pt`` files.

        The VLM backbone is not saved — reload it from HuggingFace Hub.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, app in self.appendages.items():
            torch.save(app.state_dict(), path / f"appendage_{name}.pt")
        meta = {
            "appendage_names": list(self.appendages.keys()),
            "appendage_types": {n: type(a).__name__ for n, a in self.appendages.items()},
            "hidden_dim": self.config.hidden_dim,
            "feature_extraction": self.config.feature_extraction,
            "n_params_total": sum(
                p.numel() for a in self.appendages.values() for p in a.parameters()
            ),
        }
        (path / "composite_config.json").write_text(json.dumps(meta, indent=2))
        print(f"[CompositeGraft] Saved {len(self.appendages)} appendages → {path}")

    def load_appendages(self, path: str | Path, strict: bool = True) -> None:
        """Load all appendage weights from a previous save() directory."""
        path = Path(path)
        for name, app in self.appendages.items():
            ckpt = path / f"appendage_{name}.pt"
            app.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=strict)
        print(f"[CompositeGraft] Loaded appendages from {path}")

    # ------------------------------------------------------------------ #
    #  Internal helpers (mirrored from VLAGraft)                          #
    # ------------------------------------------------------------------ #

    def _extract_features(self, hidden_states: torch.Tensor,
                           attention_mask: torch.Tensor | None) -> torch.Tensor:
        mode = self.config.feature_extraction
        if mode == "last":
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1
                idx = lengths.clamp(min=0).long()
                batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
                return hidden_states[batch_idx, idx]
            return hidden_states[:, -1]
        elif mode == "first":
            return hidden_states[:, 0]
        elif mode == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)
            return hidden_states.mean(1)
        raise ValueError(f"Unknown feature_extraction mode: {mode!r}")

    def _detect_hidden_dim(self) -> int:
        cfg = self.vlm.config
        for attr in ("hidden_size", "d_model", "n_embd", "dim", "embed_dim"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            for attr in ("hidden_size", "d_model", "n_embd"):
                if hasattr(text_cfg, attr):
                    return int(getattr(text_cfg, attr))
        raise ValueError("Cannot auto-detect hidden_dim from model config.")

    def _find_vision_encoder(self) -> nn.Module | None:
        _CANDIDATE_PATHS = [
            "model.vision_model", "vision_model", "visual",
            "vision_tower", "vision_encoder", "model.visual",
        ]
        for path in _CANDIDATE_PATHS:
            module = self.vlm
            try:
                for part in path.split("."):
                    module = getattr(module, part)
                if isinstance(module, nn.Module):
                    return module
            except AttributeError:
                continue
        return None

    def __repr__(self) -> str:
        n_vlm = sum(p.numel() for p in self.vlm.parameters())
        n_app = sum(p.numel() for a in self.appendages.values() for p in a.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head_str = "\n".join(
            f"    {name}: {type(app).__name__} ({sum(p.numel() for p in app.parameters()):,} params)"
            for name, app in self.appendages.items()
        )
        return (
            f"CompositeGraft(\n"
            f"  vlm={type(self.vlm).__name__} ({n_vlm:,} params)\n"
            f"  appendages=\n{head_str}\n"
            f"  trainable={n_trainable:,} params\n"
            f"  feature_extraction={self.config.feature_extraction!r}\n"
            f")"
        )
