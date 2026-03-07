#!/usr/bin/env python3
"""
Train a CompositeGraft — multiple appendages sharing one VLM forward pass.

Default combination: JoystickAppendage + ButtonAppendage on FruitCatcherEnvironment.
This demonstrates how one vision-language backbone can simultaneously drive
two distinct action modalities.

Usage:
    python scripts/train_composite.py
    python scripts/train_composite.py --combo joystick+button --bc-steps 800
    python scripts/train_composite.py --combo dpad+button --env treasure_hunt
    python scripts/train_composite.py --device cuda --save-dir runs/composite_v1
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from vla_hands.appendages.button import ButtonAppendage, MultiButtonAppendage
from vla_hands.appendages.dpad import DPadAppendage
from vla_hands.appendages.joystick import JoystickAppendage
from vla_hands.environments.fruit_catcher import FruitCatcherEnvironment
from vla_hands.environments.mcq_navigator import MCQNavigatorEnvironment
from vla_hands.environments.treasure_hunt import TreasureHuntEnvironment
from vla_hands.grafting.composite import CompositeGraft
from vla_hands.grafting.graft import GraftConfig


COMBOS = {
    "joystick+button": (
        lambda hd: {"joystick": JoystickAppendage(hd), "button": ButtonAppendage(hd)},
        "FruitCatcher",
    ),
    "dpad+button": (
        lambda hd: {"dpad": DPadAppendage(hd), "button": ButtonAppendage(hd)},
        "TreasureHunt",
    ),
    "dpad+multibutton": (
        lambda hd: {"dpad": DPadAppendage(hd), "choice": MultiButtonAppendage(hd, n_buttons=4, labels=["A","B","C","D"])},
        "MCQNavigator",
    ),
}

ENV_MAP = {
    "FruitCatcher": FruitCatcherEnvironment,
    "TreasureHunt": TreasureHuntEnvironment,
    "MCQNavigator": MCQNavigatorEnvironment,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a CompositeGraft with multiple appendages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument("--combo", default="joystick+button", choices=list(COMBOS))
    p.add_argument("--bc-steps", type=int, default=600)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--save-dir", default="model_checkpoints/composite")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hidden_dim(vlm) -> int:
    text_cfg = getattr(vlm.config, "text_config", None)
    return (
        text_cfg.hidden_size
        if text_cfg is not None and hasattr(text_cfg, "hidden_size")
        else vlm.config.hidden_size
    )


def _bc_step(graft, batch, device):
    """Single joint BC update. Returns total loss (sum over appendages)."""
    obs_batch, action_batch = batch
    total_loss = torch.tensor(0.0, device=device)
    out = graft(**obs_batch)
    for name, appendage in graft.appendages.items():
        if name not in action_batch:
            continue
        pred = out[name]
        tgt = action_batch[name].to(device)
        total_loss = total_loss + appendage.action_loss(pred, tgt)
    return total_loss


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    combo_fn, default_env = COMBOS[args.combo]

    print(f"Device:   {device}")
    print(f"Model:    {args.model}")
    print(f"Combo:    {args.combo}  →  {default_env}")
    print(f"BC steps: {args.bc_steps}")
    print()

    # ── Load VLM ─────────────────────────────────────────────────────────
    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(args.model)
    vlm = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32)
    print(f"  Parameters: {sum(p.numel() for p in vlm.parameters()):,}")

    hd = _hidden_dim(vlm)
    appendages = combo_fn(hd)

    graft = CompositeGraft(
        vlm=vlm,
        appendages=appendages,
        config=GraftConfig(feature_extraction="last"),
    )
    print(f"CompositeGraft: {list(appendages.keys())}")

    # ── Environment ──────────────────────────────────────────────────────
    env = ENV_MAP[default_env]()
    print(f"Environment:  {type(env).__name__}")

    # ── Minimal BC training loop ─────────────────────────────────────────
    # (For a full curriculum, use auto_curriculum() from vla_hands.training.auto)
    from vla_hands.training.trainer import _preprocess

    optimizer = torch.optim.AdamW(graft.parameters(), lr=3e-4)
    graft.to(device)
    graft.train()

    print(f"\nBC Training ({args.bc_steps} steps)...")
    for step in range(args.bc_steps):
        obs = env.reset(seed=step)
        expert = env.expert_action()  # dict: {"joystick": [...], "button": ...}

        inputs = _preprocess(processor, obs, env.prompt, device)
        out = graft(**inputs)

        total_loss = torch.tensor(0.0, device=device)
        for name, appendage in graft.appendages.items():
            if isinstance(expert, dict) and name in expert:
                import numpy as np

                raw = expert[name]
                if isinstance(raw, (int, float, bool)):
                    tgt = torch.tensor([[float(raw)]], device=device)
                else:
                    tgt = torch.tensor([list(raw)], dtype=torch.float32, device=device)
                pred = out[name]
                # Align shapes
                if pred.shape != tgt.shape:
                    tgt = tgt.view_as(pred)
                total_loss = total_loss + appendage.action_loss(pred, tgt)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(graft.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % max(1, args.bc_steps // 10) == 0:
            print(f"  step {step+1:4d}/{args.bc_steps}  loss={total_loss.item():.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = Path(args.save_dir)
    graft.save(save_path)
    print(f"\nSaved to {save_path}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
