"""Training environments for VLA appendages."""

from .base import BaseEnvironment, EnvStepResult
from .button_task import ButtonPressEnvironment, MCQButtonEnvironment
from .grid_world import GridWorldEnvironment
from .maze import MazeEnvironment
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
]
