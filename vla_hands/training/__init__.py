"""Training infrastructure for VLA grafts."""

from .trainer import BCTrainer, CurriculumConfig, RLTrainer, TrainerConfig, TrainingCurriculum
from .auto import (
    make_env,
    make_appendage,
    make_graft,
    auto_curriculum,
    recommended_envs,
    APPENDAGE_ENV_MAP,
)

__all__ = [
    "BCTrainer",
    "RLTrainer",
    "TrainerConfig",
    "CurriculumConfig",
    "TrainingCurriculum",
    "make_env",
    "make_appendage",
    "make_graft",
    "auto_curriculum",
    "recommended_envs",
    "APPENDAGE_ENV_MAP",
]
