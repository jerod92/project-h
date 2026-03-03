"""Training environments for VLA appendages."""

from .base import BaseEnvironment, EnvStepResult
from .button_task import ButtonPressEnvironment, MCQButtonEnvironment
from .grid_world import GridWorldEnvironment
from .maze import MazeEnvironment
from .pointing import PointingEnvironment
from .prompt_vocab import (
    BUTTON_PRESS_VOCAB,
    GRID_WORLD_VOCAB,
    MAZE_VOCAB,
    MCQ_VOCAB,
    POINTING_VOCAB,
    SPACESHIP_VOCAB,
    TARGET_NAV_VOCAB,
    PromptVocab,
)
from .spaceship import SpaceshipNavEnvironment
from .target_nav import TargetNavEnvironment

__all__ = [
    "BaseEnvironment",
    "EnvStepResult",
    # Joystick
    "TargetNavEnvironment",
    "SpaceshipNavEnvironment",
    # D-pad
    "GridWorldEnvironment",
    "MazeEnvironment",
    # Button
    "ButtonPressEnvironment",
    "MCQButtonEnvironment",
    # Touchscreen / pointing
    "PointingEnvironment",
    # Prompt vocab
    "PromptVocab",
    "TARGET_NAV_VOCAB",
    "SPACESHIP_VOCAB",
    "GRID_WORLD_VOCAB",
    "MAZE_VOCAB",
    "BUTTON_PRESS_VOCAB",
    "MCQ_VOCAB",
    "POINTING_VOCAB",
]
