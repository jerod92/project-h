"""
TrainingCurriculum — orchestrates the full BC → RL training pipeline.

Provides a single high-level entry point that sequences:
  1. Behavioral Cloning (BCTrainer): warm-start the action head from expert demos
  2. RL Fine-tuning (RLTrainer): optimize for environment reward

This two-phase approach is robust: BC gives a non-random starting policy,
which dramatically reduces the variance of the subsequent REINFORCE training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.optim as optim

from ..environments.base import BaseEnvironment
from ..grafting.freezing import DEFAULT_CURRICULUM, FreezingStage
from ..grafting.graft import VLAGraft
from .trainer import BCTrainer, RLTrainer, TrainerConfig


@dataclass
class CurriculumConfig:
    """Top-level curriculum configuration."""

    # BC phase
    bc_steps: int = 1000
    bc_batch_size: int = 4

    # RL phase (set to 0 to skip)
    rl_steps: int = 500
    rl_episodes_per_update: int = 4

    # Shared
    appendage_lr: float = 1e-4
    vlm_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Logging / saving
    log_every: int = 50
    eval_every: int = 200
    eval_episodes: int = 5
    save_every: int = 500
    save_dir: str = "model_checkpoints"

    # Device
    device: str = "cpu"

    # Freezing curriculum
    freezing_stages: list[FreezingStage] = field(
        default_factory=lambda: DEFAULT_CURRICULUM
    )

    def to_trainer_config(self) -> TrainerConfig:
        return TrainerConfig(
            appendage_lr=self.appendage_lr,
            vlm_lr=self.vlm_lr,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            bc_steps=self.bc_steps,
            batch_size=self.bc_batch_size,
            rl_steps=self.rl_steps,
            episodes_per_update=self.rl_episodes_per_update,
            log_every=self.log_every,
            eval_every=self.eval_every,
            eval_episodes=self.eval_episodes,
            save_every=self.save_every,
            save_dir=self.save_dir,
            freezing_stages=self.freezing_stages,
        )


class TrainingCurriculum:
    """
    End-to-end training pipeline for a VLA graft.

    Example::

        graft = VLAGraft(vlm=model, appendage=JoystickAppendage(hidden_dim))
        env   = TargetNavEnvironment()
        curriculum = TrainingCurriculum(graft, processor, env, CurriculumConfig())
        results = curriculum.run()
    """

    def __init__(
        self,
        graft: VLAGraft,
        processor,
        environment: BaseEnvironment,
        config: CurriculumConfig | None = None,
    ):
        self.graft = graft
        self.processor = processor
        self.env = environment
        self.config = config or CurriculumConfig()

    def run(self) -> dict[str, list[dict]]:
        """
        Execute the full BC → RL curriculum.

        Returns:
            dict with "bc" and "rl" keys mapping to lists of per-step metrics.
        """
        trainer_cfg = self.config.to_trainer_config()
        device = self.config.device

        # ── Phase 1: Behavioral Cloning ──────────────────────────────────
        bc_trainer = BCTrainer(
            graft=self.graft,
            processor=self.processor,
            environment=self.env,
            config=trainer_cfg,
            device=device,
        )
        bc_metrics = bc_trainer.train()

        # ── Phase 2: RL Fine-tuning ──────────────────────────────────────
        rl_metrics: list[dict] = []
        if self.config.rl_steps > 0:
            rl_trainer = RLTrainer(
                graft=self.graft,
                processor=self.processor,
                environment=self.env,
                config=trainer_cfg,
                device=device,
            )
            rl_optimizer = optim.Adam(
                self.graft.appendage.parameters(),
                lr=self.config.appendage_lr * 0.1,
            )
            rl_metrics = rl_trainer.train(optimizer=rl_optimizer)

        return {"bc": bc_metrics, "rl": rl_metrics}
