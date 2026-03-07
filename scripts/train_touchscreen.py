#!/usr/bin/env python3
"""
Train a touchscreen appendage on the pointing environment.

The TouchscreenAppendage predicts absolute (x, y) tap coordinates ∈ [0, 1]².
An optional vision skip connection feeds patch embeddings directly from the
vision encoder to the prediction head — useful for spatial precision tasks.

Usage:
    python scripts/train_touchscreen.py
    python scripts/train_touchscreen.py --vision-skip --bc-steps 600
    python scripts/train_touchscreen.py --device cuda --save-dir runs/ts_v1
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from vla_hands import (
    CurriculumConfig,
    GraftConfig,
    TrainingCurriculum,
    VLAGraft,
    run_expert_baseline,
)
from vla_hands.appendages.touchscreen import TouchscreenAppendage
from vla_hands.utils.benchmark import BenchmarkSuite
from vla_hands.environments.pointing import PointingEnvironment
from vla_hands.grafting.freezing import DEFAULT_CURRICULUM, QUICK_CURRICULUM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a touchscreen VLA appendage on the pointing environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument("--bc-steps", type=int, default=500)
    p.add_argument("--rl-steps", type=int, default=0, help="0 = BC only (recommended for pointing)")
    p.add_argument("--rl-max-steps", type=int, default=30, help="Max steps per RL episode.")
    p.add_argument("--ppo-clip", type=float, default=0.2, help="PPO clip ratio ε.")
    p.add_argument("--ppo-epochs", type=int, default=4, help="PPO gradient epochs per rollout batch.")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE smoothing parameter λ.")
    p.add_argument("--action-std", type=float, default=0.3, help="Std for continuous action distributions.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--save-dir", default="model_checkpoints/touchscreen")
    p.add_argument("--env-size", type=int, default=256)
    p.add_argument("--n-distractors", type=int, default=3, help="Number of distractor circles.")
    p.add_argument("--n-colors", type=int, default=5)
    p.add_argument("--vision-skip", action="store_true", help="Enable vision encoder skip connection.")
    p.add_argument("--eval-episodes", type=int, default=15)
    p.add_argument("--curriculum", default="quick", choices=["quick", "default"])
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

    print(f"Device:         {device}")
    print(f"Model:          {args.model}")
    print(f"BC steps:       {args.bc_steps}")
    print(f"RL steps:       {args.rl_steps}")
    print(f"Vision skip:    {args.vision_skip}")
    print(f"Distractors:    {args.n_distractors}")
    print()

    # ── Load VLM ─────────────────────────────────────────────────────────
    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(args.model)
    vlm = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32)
    print(f"  Parameters: {sum(p.numel() for p in vlm.parameters()):,}")

    text_cfg = getattr(vlm.config, "text_config", None)
    hidden_dim = (
        text_cfg.hidden_size
        if text_cfg is not None and hasattr(text_cfg, "hidden_size")
        else vlm.config.hidden_size
    )

    vision_dim = VLAGraft.detect_vision_dim(vlm) if args.vision_skip else None
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Vision dim:   {vision_dim if vision_dim else 'N/A (no skip)'}")

    # ── Build appendage + graft ──────────────────────────────────────────
    appendage = TouchscreenAppendage(
        hidden_dim=hidden_dim,
        vision_dim=vision_dim,
    )
    graft = VLAGraft(
        vlm=vlm,
        appendage=appendage,
        config=GraftConfig(feature_extraction="last"),
    )
    print(graft)
    hook_status = "registered" if graft._vision_hook is not None else "not registered"
    print(f"  Vision hook:  {hook_status}")

    # ── Environment ──────────────────────────────────────────────────────
    env = PointingEnvironment(
        width=args.env_size,
        height=args.env_size,
        n_distractors=args.n_distractors,
        n_colors=args.n_colors,
    )

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
    curriculum = TrainingCurriculum(graft, processor, env, config)
    curriculum.run()

    # ── Final benchmark ──────────────────────────────────────────────────
    print("\nRunning final benchmark...")
    suite = BenchmarkSuite(graft, processor, [env], device=device)
    results = suite.run_all(n_episodes=args.eval_episodes)
    print(results[0])

    return 0


if __name__ == "__main__":
    sys.exit(main())
