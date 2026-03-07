"""Grafting system: attaches action appendages to VLMs."""

from .freezing import (
    DEFAULT_CURRICULUM,
    QUICK_CURRICULUM,
    STAGE_APPENDAGE_ONLY,
    STAGE_FULL,
    STAGE_LAST_2,
    STAGE_LAST_6,
    FreezingCurriculum,
    FreezingStage,
)
from .graft import GraftConfig, VLAGraft
from .composite import CompositeGraft
from .vision_bridge import VisionBridge
from .lora import LoRAConfig, apply_lora, lora_parameter_count, merge_lora

__all__ = [
    "GraftConfig",
    "VLAGraft",
    "CompositeGraft",
    "VisionBridge",
    "FreezingCurriculum",
    "FreezingStage",
    "DEFAULT_CURRICULUM",
    "QUICK_CURRICULUM",
    "STAGE_APPENDAGE_ONLY",
    "STAGE_LAST_2",
    "STAGE_LAST_6",
    "STAGE_FULL",
    # LoRA
    "LoRAConfig",
    "apply_lora",
    "merge_lora",
    "lora_parameter_count",
]
