#!/usr/bin/env python3
"""
Train a joystick appendage on the target navigation environment.

Usage:
    python scripts/train_joystick.py
    python scripts/train_joystick.py --model HuggingFaceTB/SmolVLM-Instruct --bc-steps 1000
    python scripts/train_joystick.py --device cuda --save-dir runs/joystick_v1
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from vla_hands import (
    CurriculumConfig,
    GraftConfig,
    JoystickAppendage,
    TargetNavEnvironment,
    TrainingCurriculum,
    VLAGraft,
    run_expert_baseline,
)
from vla_hands.benchmarks.suite import BenchmarkSuite
from vla_hands.grafting.freezing import QUICK_CURRICULUM, DEFAULT_CURRICULUM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a joystick VLA appendage via BC + RL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM-Instruct",
        help="HuggingFace model ID for the base VLM.",
    )
    p.add_argument("--bc-steps", type=int, default=500, help="Behavioral cloning steps.")
    p.add_argument("--rl-steps", type=int, default=200, help="RL fine-tuning steps (0 to skip).")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device.",
    )
    p.add_argument("--save-dir", default="checkpoints/joystick", help="Checkpoint directory.")
    p.add_argument("--env-size", type=int, default=224, help="Environment image size (px).")
    p.add_argument("--eval-episodes", type=int, default=10, help="Episodes for final benchmark.")
    p.add_argument(
        "--curriculum",
        default="default",
        choices=["quick", "default"],
        help="Freezing curriculum: 'quick' = appendage only, 'default' = gradual unfreeze.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
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
    print(f"Device:       {device}")
    print(f"Model:        {args.model}")
    print(f"BC steps:     {args.bc_steps}")
    print(f"RL steps:     {args.rl_steps}")
    print(f"Save dir:     {args.save_dir}")
    print()

    # ── Load VLM ─────────────────────────────────────────────────────────
    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(args.model)
    vlm = AutoModelForVision2Seq.from_pretrained(args.model, torch_dtype=torch.float32)
    print(f"  Parameters: {sum(p.numel() for p in vlm.parameters()):,}")

    # ── Build appendage + graft ──────────────────────────────────────────
    hidden_dim = vlm.config.hidden_size
    appendage = JoystickAppendage(hidden_dim=hidden_dim)
    graft = VLAGraft(
        vlm=vlm,
        appendage=appendage,
        config=GraftConfig(feature_extraction="last"),
    )
    print(graft)

    # ── Environment ──────────────────────────────────────────────────────
    env = TargetNavEnvironment(
        width=args.env_size,
        height=args.env_size,
        max_steps=100,
    )

    # Expert baseline (upper bound)
    print("\nRunning expert baseline...")
    run_expert_baseline(env, n_episodes=args.eval_episodes)

    # ── Curriculum ───────────────────────────────────────────────────────
    stages = QUICK_CURRICULUM if args.curriculum == "quick" else DEFAULT_CURRICULUM
    config = CurriculumConfig(
        bc_steps=args.bc_steps,
        rl_steps=args.rl_steps,
        save_dir=args.save_dir,
        device=device,
        freezing_stages=stages,
    )
    curriculum = TrainingCurriculum(
        graft=graft,
        processor=processor,
        environment=env,
        config=config,
    )
    curriculum.run()

    # ── Final benchmark ──────────────────────────────────────────────────
    print("\nRunning final benchmark...")
    suite = BenchmarkSuite(
        graft=graft,
        processor=processor,
        environments=[env],
        device=device,
    )
    results = suite.run_all(n_episodes=args.eval_episodes)
    print(results[0])

    return 0


if __name__ == "__main__":
    sys.exit(main())
