#!/usr/bin/env python3
"""
Auto-curriculum training — one command for any appendage combination.

The auto library automatically picks the best environment(s), builds the
right graft (single or composite), and runs BC + RL training.

Usage:
    python scripts/train_auto.py --appendages joystick
    python scripts/train_auto.py --appendages joystick button
    python scripts/train_auto.py --appendages dpad multibutton --budget 1500
    python scripts/train_auto.py --appendages touchscreen --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from vla_hands.training.auto import auto_curriculum, recommended_envs

VALID_APPENDAGES = ["joystick", "dpad", "button", "multibutton", "touchscreen"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-curriculum VLA training for any appendage combination.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--appendages",
        nargs="+",
        choices=VALID_APPENDAGES,
        required=True,
        help="One or more appendage names to train together.",
    )
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument("--budget", type=int, default=1000, help="Total training steps (BC + RL).")
    p.add_argument("--rl-max-steps", type=int, default=30, help="Max steps per RL episode.")
    p.add_argument("--ppo-clip", type=float, default=0.2, help="PPO clip ratio ε.")
    p.add_argument("--ppo-epochs", type=int, default=4, help="PPO gradient epochs per rollout batch.")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE smoothing parameter λ.")
    p.add_argument("--action-std", type=float, default=0.3, help="Std for continuous action distributions.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--save-dir", default=None, help="Checkpoint directory (auto-named if omitted).")
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


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    combo_str = "+".join(args.appendages)
    save_dir = args.save_dir or f"model_checkpoints/auto_{combo_str}"

    print(f"Device:      {device}")
    print(f"Model:       {args.model}")
    print(f"Appendages:  {args.appendages}")
    print(f"Budget:      {args.budget} steps")
    print(f"Save dir:    {save_dir}")
    print()

    # Show recommended environments before loading
    recs = recommended_envs(args.appendages)
    if recs:
        print("Recommended environments:")
        for env_name, desc in recs[:3]:
            print(f"  {env_name}: {desc}")
    print()

    # ── Load VLM ─────────────────────────────────────────────────────────
    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(args.model)
    vlm = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32)
    print(f"  Parameters: {sum(p.numel() for p in vlm.parameters()):,}")
    print()

    # ── Auto curriculum ──────────────────────────────────────────────────
    graft, metrics = auto_curriculum(
        vlm=vlm,
        processor=processor,
        appendage_names=args.appendages,
        budget_steps=args.budget,
        rl_max_steps_per_episode=args.rl_max_steps,
        rl_ppo_clip=args.ppo_clip,
        rl_ppo_epochs=args.ppo_epochs,
        rl_gae_lambda=args.gae_lambda,
        rl_action_std=args.action_std,
        device=device,
        save_dir=save_dir,
    )

    print("\nTraining complete!")
    print(f"Graft type: {type(graft).__name__}")
    print(f"Checkpoint: {save_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
