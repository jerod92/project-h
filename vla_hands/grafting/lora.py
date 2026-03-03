"""
LoRA integration for VLA graft training.

Low-Rank Adaptation (LoRA) adds small trainable rank-decomposition matrices
to the VLM's attention/MLP layers.  This is dramatically more efficient than
full fine-tuning:

  - Only ~0.1–1% of parameters are trained
  - Memory footprint is ~5–10× lower (no optimizer states for frozen params)
  - Training speed is ~3–5× faster for the VLM backbone steps
  - Catastrophic forgetting risk is much lower

LoRA is applied to the VLM *before* wrapping it in VLAGraft.  The resulting
PEFT model is otherwise drop-in compatible with VLAGraft — the forward pass
interface is unchanged.

Usage
-----
    from vla_hands.grafting.lora import LoRAConfig, apply_lora, merge_lora

    lora_cfg = LoRAConfig(r=8, alpha=16, dropout=0.05)
    vlm = apply_lora(vlm, lora_cfg)

    graft = VLAGraft(vlm, appendage)
    # Train as normal ...

    # Optionally fuse LoRA weights into the base model (for deployment):
    vlm = merge_lora(vlm)

Saving
------
LoRA adapters can be saved compactly (only the low-rank matrices):

    vlm.save_pretrained("my_lora_adapter/")

And loaded back on top of the frozen base model:

    from peft import PeftModel
    vlm = PeftModel.from_pretrained(base_vlm, "my_lora_adapter/")

Requires
--------
    pip install peft>=0.10.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    pass  # avoid circular imports


# ── LoRAConfig ────────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adaptation of a VLM backbone.

    Parameters follow the PEFT library conventions.
    See: https://huggingface.co/docs/peft/conceptual_guides/lora
    """

    # Rank of the low-rank decomposition matrices (higher = more expressive).
    # Typical values: 4 (minimal), 8 (default), 16 (expressive), 32 (large).
    r: int = 8

    # Scaling factor: effective LR scale ≈ alpha / r.
    # Keeping alpha = 2*r is a common heuristic.
    alpha: float = 16.0

    # Dropout probability applied to LoRA inputs (regularisation).
    dropout: float = 0.05

    # Which linear modules to adapt.  "auto" tries a sensible default for the
    # detected model family.  Pass a list to override explicitly, e.g.:
    #   ["q_proj", "v_proj"]   — attention only (cheapest, often enough)
    #   ["q_proj", "k_proj", "v_proj", "o_proj"]   — full attention
    #   ["q_proj", "v_proj", "gate_proj", "up_proj"]  — attention + FFN
    target_modules: list[str] | str = "auto"

    # Whether to adapt the vision encoder layers (more expensive, rarely needed
    # for short fine-tuning runs; set True if you have a GPU with 24 GB+).
    adapt_vision_encoder: bool = False

    # Task type string for PEFT (keep as "CAUSAL_LM" for decoder VLMs).
    task_type: str = "CAUSAL_LM"

    # Modules to *exclude* from LoRA even if they match target_modules.
    # By default the language model head is excluded.
    modules_to_save: list[str] = field(default_factory=list)


# ── Default target modules per architecture family ────────────────────────────

_FAMILY_TARGETS: dict[str, list[str]] = {
    # LLaMA / Mistral / Qwen style
    "llama":   ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "qwen":    ["q_proj", "v_proj"],
    # Phi
    "phi":     ["q_proj", "v_proj"],
    # Idefics3 / SmolVLM text part uses LLaMA-style layers
    "idefics": ["q_proj", "v_proj"],
    # Fallback — works for most HF transformer decoders
    "default": ["q_proj", "v_proj"],
}


def _resolve_target_modules(vlm: nn.Module, cfg: LoRAConfig) -> list[str]:
    """
    Determine which module names to adapt.

    Inspects vlm.config.model_type (or class name) to pick a sensible default
    when cfg.target_modules == "auto".
    """
    if cfg.target_modules != "auto":
        return list(cfg.target_modules)  # type: ignore[arg-type]

    model_type: str = ""
    try:
        model_type = vlm.config.model_type.lower()
    except AttributeError:
        model_type = type(vlm).__name__.lower()

    for family, targets in _FAMILY_TARGETS.items():
        if family in model_type:
            return targets

    return _FAMILY_TARGETS["default"]


# ── Public API ────────────────────────────────────────────────────────────────

def apply_lora(vlm: nn.Module, config: LoRAConfig | None = None) -> nn.Module:
    """
    Wrap *vlm* with LoRA adapters and return the PEFT model.

    Args:
        vlm:    A HuggingFace model (e.g. returned by AutoModel.from_pretrained).
        config: LoRA configuration.  Uses sensible defaults if None.

    Returns:
        A ``peft.PeftModel`` that is otherwise interface-compatible with the
        original model but has far fewer trainable parameters.

    Raises:
        ImportError: If the ``peft`` library is not installed.
    """
    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    except ImportError as exc:
        raise ImportError(
            "The 'peft' library is required for LoRA training.\n"
            "Install it with:  pip install peft>=0.10.0"
        ) from exc

    cfg = config or LoRAConfig()

    target_modules = _resolve_target_modules(vlm, cfg)

    # Build layers_to_transform / layers_pattern to skip vision encoder
    # if adapt_vision_encoder is False (default).  This is tricky to do
    # generically, so we rely on target_modules naming conventions — vision
    # encoder layers typically live under "vision_model.*" and their linear
    # layers are named differently (e.g. "attn.proj", "mlp.fc1").
    # By only specifying language-decoder layer names, the vision encoder is
    # naturally excluded.

    peft_cfg = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=target_modules,
        modules_to_save=cfg.modules_to_save or None,
        bias="none",
    )

    peft_model = get_peft_model(vlm, peft_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def merge_lora(vlm: nn.Module) -> nn.Module:
    """
    Merge LoRA adapter weights into the base model and return an unmodified
    ``nn.Module`` with no PEFT overhead.

    Call this before deployment / when saving a non-PEFT checkpoint.

    Args:
        vlm: A ``peft.PeftModel`` previously created by ``apply_lora()``.

    Returns:
        The base model with LoRA weights fused in.
    """
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "The 'peft' library is required to merge LoRA weights.\n"
            "Install it with:  pip install peft>=0.10.0"
        ) from exc

    if not isinstance(vlm, PeftModel):
        raise TypeError(
            f"Expected a peft.PeftModel, got {type(vlm).__name__}. "
            "Apply LoRA with apply_lora() first."
        )

    merged = vlm.merge_and_unload()
    print("[LoRA] Adapter weights merged into base model.")
    return merged


def lora_parameter_count(vlm: nn.Module) -> dict[str, int]:
    """
    Return a summary of trainable vs total parameters for a LoRA-wrapped model.

    Returns:
        dict with keys "trainable", "total", "pct_trainable".
    """
    trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vlm.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "pct_trainable": 100.0 * trainable / max(total, 1),
    }
