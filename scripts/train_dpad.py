#!/usr/bin/env python3
"""
Train a D-pad appendage on the grid-world navigation environment.

Usage:
    python scripts/train_dpad.py
    python scripts/train_dpad.py --model HuggingFaceTB/SmolVLM-Instruct --bc-steps 1000
    python scripts/train_dpad.py --grid-size 6 --device cuda --save-dir runs/dpad_v1
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from vla_hands import (
    CurriculumConfig,
    DPadAppendage,
    GraftConfig,
    GridWorldEnvironment,
    TrainingCurriculum,
    VLAGraft,
    run_expert_baseline,
)
from vla_hands.benchmarks.suite import BenchmarkSuite
from vla_hands.grafting.freezing import DEFAULT_CURRICULUM, QUICK_CURRICULUM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a D-pad VLA appendage on grid-world navigation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM-Instruct",
        help="HuggingFace model ID for the base VLM.",
    )
    p.add_argument("--bc-steps", type=int, default=500, help="Behavioral cloning steps.")
    p.add_argument("--rl-steps", type=int, default=200, help="RL fine-tuning steps (0 to skip). Each step = rl_episodes_per_update full episodes.")
    p.add_argument("--rl-max-steps", type=int, default=30, help="Max steps per RL episode (shorter = faster).")
    p.add_argument("--ppo-clip", type=float, default=0.2, help="PPO clip ratio ε.")
    p.add_argument("--ppo-epochs", type=int, default=4, help="PPO gradient epochs per rollout batch.")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE smoothing parameter λ.")
    p.add_argument("--action-std", type=float, default=0.3, help="Std for continuous action distributions.")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device.",
    )
    p.add_argument("--save-dir", default="model_checkpoints/dpad", help="Checkpoint directory.")
    p.add_argument("--grid-size", type=int, default=8, help="Grid world size (NxN cells).")
    p.add_argument("--wall-density", type=float, default=0.20, help="Fraction of cells that are walls.")
    p.add_argument("--eval-episodes", type=int, default=10, help="Episodes for final benchmark.")
    p.add_argument(
        "--curriculum",
        default="default",
        choices=["quick", "default"],
        help="Freezing curriculum.",
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
    print(f"Grid size:    {args.grid_size}x{args.grid_size}")
    print(f"Wall density: {args.wall_density:.0%}")
    print(f"Save dir:     {args.save_dir}")
    print()

    # ── Load VLM ─────────────────────────────────────────────────────────
    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(args.model)
    vlm = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32)
    print(f"  Parameters: {sum(p.numel() for p in vlm.parameters()):,}")

    # ── Build appendage + graft ──────────────────────────────────────────
    text_cfg = getattr(vlm.config, "text_config", None)
    hidden_dim = (
        text_cfg.hidden_size
        if text_cfg is not None and hasattr(text_cfg, "hidden_size")
        else vlm.config.hidden_size
    )
    appendage = DPadAppendage(hidden_dim=hidden_dim)
    graft = VLAGraft(
        vlm=vlm,
        appendage=appendage,
        config=GraftConfig(feature_extraction="last"),
    )
    print(graft)

    # ── Environment ──────────────────────────────────────────────────────
    env = GridWorldEnvironment(
        grid_size=args.grid_size,
        wall_density=args.wall_density,
        max_steps=args.grid_size * args.grid_size * 2,
    )

    # Expert baseline
    print("\nRunning expert baseline...")
    run_expert_baseline(env, n_episodes=args.eval_episodes)

    # ── Curriculum ───────────────────────────────────────────────────────
    stages = QUICK_CURRICULUM if args.curriculum == "quick" else DEFAULT_CURRICULUM
    config = CurriculumConfig(
        bc_steps=args.bc_steps,
        rl_steps=args.rl_steps,
        rl_max_steps_per_episode=args.rl_max_steps,
        rl_ppo_clip=args.ppo_clip,
        rl_ppo_epochs=args.ppo_epochs,
        rl_gae_lambda=args.gae_lambda,
        rl_action_std=args.action_std,
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
