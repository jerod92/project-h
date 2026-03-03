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
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..appendages.base import BaseAppendage


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
