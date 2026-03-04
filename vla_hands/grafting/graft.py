"""
VLAGraft — the core class for attaching action appendages to a VLM.

Design decisions answered here:

Q: Output timing — does the action head fire every forward pass?
A: Yes. On every forward pass that includes pixel_values (vision input), the action
   head fires in parallel with token generation. The action head uses the hidden state
   of the LAST non-padding token position as its input feature vector. This means:
     - Every image observation produces an action prediction
     - The language decoding path is untouched; the model continues generating tokens
   In practice, for RL episodes the caller discards token logits and only uses action.
   For language tasks, the caller discards the action. The head is always "on" —
   it just costs a few extra FLOPS (the MLP is tiny compared to the VLM).

Q: Which hidden state to use?
A: The final transformer layer's hidden state at the last non-padding token.
   This position has attended over all image tokens AND all text tokens, giving
   the richest semantic signal for action prediction. Alternatives (first token,
   mean pooling) are configurable via GraftConfig.feature_extraction.

Q: Model compatibility?
A: Any HuggingFace model that accepts pixel_values + input_ids and supports
   output_hidden_states=True. Tested families: LLaVA, SmolVLM, Qwen-VL, InstructBLIP.
   The VLM backbone is never structurally modified — we only add the appendage head.

Q: Saving?
A: VLM weights are saved separately (via HuggingFace save_pretrained).
   Only the appendage weights are saved in the graft checkpoint, along with a
   JSON config describing the graft. This keeps checkpoints small (MBs, not GBs).

Q: Vision skip connections for TouchscreenAppendage?
A: When the attached appendage has needs_vision_features=True, VLAGraft registers a
   forward hook on the VLM's vision encoder.  The hook captures mean-pooled patch
   embeddings during the forward pass and passes them as vision_features to the
   appendage alongside the LLM hidden state.  This lets the action head "see" the raw
   spatial layout of the image, bypassing the LLM's information bottleneck.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..appendages.base import BaseAppendage


# ── Vision encoder forward hook ───────────────────────────────────────────────

class _VisionHook:
    """
    Forward hook that captures mean-pooled vision encoder output.

    Registered on the vision encoder sub-module of the VLM.  After the VLM's
    forward pass completes, ``self.features`` holds a [batch, vision_dim]
    tensor containing the spatially-averaged patch embeddings.

    The CLS token (index 0) is skipped so the average represents spatial
    content rather than a global summary token.
    """

    def __init__(self, module: nn.Module):
        self.features: torch.Tensor | None = None
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, inputs: Any, output: Any) -> None:
        # Unwrap various output formats from different VLM families
        if hasattr(output, "last_hidden_state"):
            feat = output.last_hidden_state     # BaseModelOutput
        elif isinstance(output, tuple):
            feat = output[0]                    # plain tuple
        else:
            feat = output                       # raw tensor

        if feat.dim() == 3:
            # [batch, n_patches, vision_dim]
            # Skip index 0 (CLS) if present; use spatial patches only
            spatial = feat[:, 1:] if feat.shape[1] > 1 else feat
            self.features = spatial.mean(dim=1).detach()   # [batch, vision_dim]
        else:
            # Unexpected shape — store as-is and let the appendage handle it
            self.features = feat.detach()

    def remove(self) -> None:
        self._handle.remove()
        self.features = None


@dataclass
class GraftConfig:
    """Configuration for a VLA graft."""

    # How to extract the action feature vector from hidden states:
    #   "last"  — hidden state at last non-padding token position (default, recommended)
    #   "first" — hidden state at position 0 (CLS-like)
    #   "mean"  — mean-pooled over non-padding tokens
    feature_extraction: str = "last"

    # Hidden dimension of the VLM (auto-detected from model.config if None)
    hidden_dim: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLAGraft(nn.Module):
    """
    Wraps a VLM + action appendage into a unified module.

    The VLM backbone is stored as self.vlm (untouched structurally).
    The action head is stored as self.appendage.

    Forward pass produces BOTH token logits (from the VLM) AND an action
    prediction (from the appendage), computed from the same hidden states.
    """

    def __init__(
        self,
        vlm: nn.Module,
        appendage: BaseAppendage,
        config: GraftConfig | None = None,
    ):
        super().__init__()
        self.vlm = vlm
        self.appendage = appendage
        self.config = config or GraftConfig()

        if self.config.hidden_dim is None:
            self.config.hidden_dim = self._detect_hidden_dim()

        # Register vision encoder hook when the appendage requests it
        self._vision_hook: _VisionHook | None = None
        if getattr(appendage, "needs_vision_features", False):
            vision_encoder = self._find_vision_encoder()
            if vision_encoder is not None:
                self._vision_hook = _VisionHook(vision_encoder)
            else:
                import warnings
                warnings.warn(
                    f"{type(appendage).__name__} has needs_vision_features=True but "
                    "VLAGraft could not locate the vision encoder sub-module. "
                    "Vision skip connection is disabled; only LLM features will be used.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------ #
    #  Forward passes                                                      #
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
        Full forward pass: VLM language modeling + action head in parallel.

        Args:
            pixel_values: Vision input [batch, C, H, W] (or model-specific format).
            input_ids: Token IDs [batch, seq].
            attention_mask: Padding mask [batch, seq].
            labels: Token labels for LM loss [batch, seq] (optional).
            **kwargs: Extra keyword arguments forwarded to the VLM.

        Returns:
            dict with:
              "lm_logits"       — [batch, seq, vocab_size]
              "action"          — [batch, *action_shape]
              "action_features" — [batch, hidden_dim]  (pre-head features)
              "lm_loss"         — scalar (only present when labels is not None)
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

        hidden_states = vlm_out.hidden_states[-1]  # [batch, seq, hidden]
        action_features = self._extract_features(hidden_states, attention_mask)

        # Pass vision skip features when the hook has captured them
        if self._vision_hook is not None and self._vision_hook.features is not None:
            action = self.appendage(action_features, vision_features=self._vision_hook.features)
        else:
            action = self.appendage(action_features)

        result: dict[str, torch.Tensor] = {
            "lm_logits": vlm_out.logits,
            "action": action,
            "action_features": action_features,
        }
        if getattr(vlm_out, "loss", None) is not None:
            result["lm_loss"] = vlm_out.loss

        return result

    @torch.no_grad()
    def predict_action(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Efficient inference: run VLM and return only the action prediction.

        Wraps the full forward pass in no_grad for evaluation / episode rollouts.
        """
        out = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return out["action"]

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def _extract_features(
        self,
        hidden_states: torch.Tensor,   # [batch, seq, hidden]
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:                  # [batch, hidden]
        mode = self.config.feature_extraction

        if mode == "last":
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1          # [batch]
                idx = lengths.clamp(min=0).long()
                batch_idx = torch.arange(
                    hidden_states.size(0), device=hidden_states.device
                )
                return hidden_states[batch_idx, idx]
            return hidden_states[:, -1]

        elif mode == "first":
            return hidden_states[:, 0]

        elif mode == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)
            return hidden_states.mean(1)

        else:
            raise ValueError(
                f"Unknown feature_extraction mode: {mode!r}. "
                "Choose from 'last', 'first', 'mean'."
            )

    # ------------------------------------------------------------------ #
    #  Optimizer helpers                                                   #
    # ------------------------------------------------------------------ #

    def parameter_groups(
        self,
        appendage_lr: float = 1e-4,
        vlm_lr: float = 1e-6,
    ) -> list[dict]:
        """
        Return AdamW-style parameter groups.

        Appendage params always have the higher LR; VLM trainable params (those
        whose requires_grad was set True by the FreezingCurriculum) use a much
        smaller LR to avoid catastrophic forgetting.
        """
        groups = [{"params": list(self.appendage.parameters()), "lr": appendage_lr}]
        vlm_params = [p for p in self.vlm.parameters() if p.requires_grad]
        if vlm_params:
            groups.append({"params": vlm_params, "lr": vlm_lr})
        return groups

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path):
        """
        Save the appendage weights and graft config to disk.

        The VLM backbone is NOT saved here — load it separately via
        HuggingFace AutoModel. Only the small appendage MLP is checkpointed.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.appendage.state_dict(), path / "appendage.pt")

        cfg = {
            **self.config.to_dict(),
            "appendage_type": type(self.appendage).__name__,
            "action_spec": {
                "name": self.appendage.action_spec.name,
                "shape": list(self.appendage.action_spec.shape),
                "dtype": self.appendage.action_spec.dtype,
                "n_discrete": self.appendage.action_spec.n_discrete,
                "low": self.appendage.action_spec.low,
                "high": self.appendage.action_spec.high,
            },
            "appendage_hidden_dim": getattr(self.appendage, "hidden_dim", None),
            "appendage_n_params": self.appendage.num_parameters(),
        }
        (path / "graft_config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[VLAGraft] Saved to {path}")

    def load_appendage(self, path: str | Path, strict: bool = True):
        """Load appendage weights from a previously saved graft."""
        path = Path(path)
        state_dict = torch.load(path / "appendage.pt", map_location="cpu")
        self.appendage.load_state_dict(state_dict, strict=strict)
        print(f"[VLAGraft] Loaded appendage from {path}")

    @classmethod
    def from_pretrained(
        cls,
        vlm_id: str,
        appendage: "BaseAppendage",
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        config: "GraftConfig | None" = None,
        vlm_kwargs: dict | None = None,
    ) -> "VLAGraft":
        """
        One-call convenience loader for inference.

        Downloads the VLM from HuggingFace Hub, loads the saved appendage
        weights, and returns a ready-to-use VLAGraft.

        Args:
            vlm_id:          HuggingFace model ID (e.g. "HuggingFaceTB/SmolVLM-256M-Instruct").
            appendage:       Pre-constructed appendage instance with correct hidden_dim.
            checkpoint_path: Directory produced by VLAGraft.save() containing
                             ``appendage.pt`` and ``graft_config.json``.
            device:          Torch device for the graft.
            config:          Override GraftConfig (auto-loaded from JSON if None).
            vlm_kwargs:      Extra kwargs forwarded to AutoModelForImageTextToText.from_pretrained.

        Returns:
            Loaded VLAGraft, moved to ``device``, in eval mode.

        Example::

            from vla_hands import VLAGraft, JoystickAppendage
            graft = VLAGraft.from_pretrained(
                vlm_id="HuggingFaceTB/SmolVLM-256M-Instruct",
                appendage=JoystickAppendage(hidden_dim=1152),
                checkpoint_path="checkpoints/bc_final",
                device="cuda",
            )
        """
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

        vlm_kwargs = vlm_kwargs or {}
        vlm = AutoModelForImageTextToText.from_pretrained(vlm_id, **vlm_kwargs)

        checkpoint_path = Path(checkpoint_path)
        if config is None:
            cfg_file = checkpoint_path / "graft_config.json"
            if cfg_file.exists():
                raw = json.loads(cfg_file.read_text())
                config = GraftConfig(
                    feature_extraction=raw.get("feature_extraction", "last"),
                    hidden_dim=raw.get("hidden_dim"),
                )

        graft = cls(vlm=vlm, appendage=appendage, config=config)
        graft.load_appendage(checkpoint_path)
        graft.to(torch.device(device))
        graft.eval()
        print(f"[VLAGraft] Ready — device={device}, appendage={type(appendage).__name__}")
        return graft

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _detect_hidden_dim(self) -> int:
        cfg = self.vlm.config
        for attr in ("hidden_size", "d_model", "n_embd", "dim", "embed_dim"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        # Try text_config for multimodal models
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            for attr in ("hidden_size", "d_model", "n_embd"):
                if hasattr(text_cfg, attr):
                    return int(getattr(text_cfg, attr))
        raise ValueError(
            "Cannot auto-detect hidden_dim from model config. "
            "Set GraftConfig(hidden_dim=...) explicitly."
        )

    def _find_vision_encoder(self) -> nn.Module | None:
        """
        Try to locate the vision encoder sub-module of the VLM.

        Checks a set of common attribute paths used by popular VLM families.
        Returns None if no vision encoder is found (graceful fallback).
        """
        # Common paths: (attribute chain as dot-separated string)
        _CANDIDATE_PATHS = [
            "model.vision_model",           # Idefics3 / SmolVLM
            "vision_model",                 # PaliGemma, BLIP-2
            "visual",                       # Qwen-VL
            "vision_tower",                 # LLaVA
            "vision_encoder",               # InstructBLIP, CogVLM
            "model.visual",                 # Qwen2-VL
            "encoder.vision_model",         # some BLIP variants
            "model.encoder.visual",
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

    @staticmethod
    def detect_vision_dim(vlm: nn.Module) -> int | None:
        """
        Attempt to read the vision encoder's hidden dimension from the model config.

        Useful for deciding the ``vision_dim`` argument to ``TouchscreenAppendage``.

        Returns:
            The integer dimension, or None if it cannot be determined.

        Example::

            vision_dim = VLAGraft.detect_vision_dim(vlm)
            head = TouchscreenAppendage(hidden_dim, vision_dim=vision_dim)
        """
        try:
            cfg = vlm.config
        except AttributeError:
            return None

        # Try direct vision_config attributes
        for vcfg_attr in ("vision_config", "vision_encoder_config", "visual_config"):
            vcfg = getattr(cfg, vcfg_attr, None)
            if vcfg is not None:
                for dim_attr in ("hidden_size", "d_model", "embed_dim"):
                    if hasattr(vcfg, dim_attr):
                        return int(getattr(vcfg, dim_attr))

        # Some models flatten vision dims directly on the top-level config
        for attr in ("vision_hidden_size", "mm_hidden_size", "visual_hidden_size"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))

        return None

    def __repr__(self) -> str:
        n_vlm = sum(p.numel() for p in self.vlm.parameters())
        n_app = self.appendage.num_parameters()
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"VLAGraft(\n"
            f"  vlm={type(self.vlm).__name__} ({n_vlm:,} params)\n"
            f"  appendage={type(self.appendage).__name__} ({n_app:,} params)\n"
            f"  trainable={n_trainable:,} params\n"
            f"  feature_extraction={self.config.feature_extraction!r}\n"
            f")"
        )
