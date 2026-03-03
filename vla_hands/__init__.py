"""
vla-hands — Give VLMs action capabilities by grafting appendage heads.

Turns Vision-Language Models (VLMs) into Vision-Language-Action models (VLAs)
by attaching lightweight action heads ("appendages") to the VLM's hidden states.

Quick start::

    from transformers import AutoProcessor, AutoModelForVision2Seq
    from vla_hands import VLAGraft, GraftConfig, JoystickAppendage
    from vla_hands import TargetNavEnvironment, TrainingCurriculum, CurriculumConfig

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    vlm = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

    appendage = JoystickAppendage(hidden_dim=vlm.config.hidden_size)
    graft = VLAGraft(vlm=vlm, appendage=appendage)

    env = TargetNavEnvironment()
    curriculum = TrainingCurriculum(graft, processor, env, CurriculumConfig(bc_steps=500))
    curriculum.run()
"""

__version__ = "0.1.0"

# Grafting
from .grafting.graft import GraftConfig, VLAGraft
from .grafting.freezing import (
    FreezingCurriculum,
    FreezingStage,
    DEFAULT_CURRICULUM,
    QUICK_CURRICULUM,
    STAGE_APPENDAGE_ONLY,
    STAGE_LAST_2,
    STAGE_LAST_6,
    STAGE_FULL,
)

# Appendages
from .appendages.base import ActionSpec, BaseAppendage
from .appendages.joystick import JoystickAppendage, JoystickAction
from .appendages.dpad import DPadAppendage, DPadButton, DPAD_DELTA

# Environments
from .environments.base import BaseEnvironment, EnvStepResult
from .environments.target_nav import TargetNavEnvironment
from .environments.grid_world import GridWorldEnvironment

# Training
from .training.trainer import BCTrainer, RLTrainer, TrainerConfig
from .training.curriculum import TrainingCurriculum, CurriculumConfig

# Benchmarks
from .benchmarks.suite import BenchmarkSuite, BenchmarkResult, compare_grafts, run_expert_baseline

__all__ = [
    # Grafting
    "VLAGraft",
    "GraftConfig",
    "FreezingCurriculum",
    "FreezingStage",
    "DEFAULT_CURRICULUM",
    "QUICK_CURRICULUM",
    "STAGE_APPENDAGE_ONLY",
    "STAGE_LAST_2",
    "STAGE_LAST_6",
    "STAGE_FULL",
    # Appendages
    "ActionSpec",
    "BaseAppendage",
    "JoystickAppendage",
    "JoystickAction",
    "DPadAppendage",
    "DPadButton",
    "DPAD_DELTA",
    # Environments
    "BaseEnvironment",
    "EnvStepResult",
    "TargetNavEnvironment",
    "GridWorldEnvironment",
    # Training
    "BCTrainer",
    "RLTrainer",
    "TrainerConfig",
    "TrainingCurriculum",
    "CurriculumConfig",
    # Benchmarks
    "BenchmarkSuite",
    "BenchmarkResult",
    "compare_grafts",
    "run_expert_baseline",
]
