"""
Base class for all VLA training environments.

Each environment:
  - Maintains internal state (positions, grid layout, etc.)
  - Renders a visual observation as a PIL Image for the VLM
  - Provides a scalar reward signal
  - Exposes an expert_action() for behavioral cloning demonstrations
  - Has a language prompt describing the task to the VLM

Environments are stateful and follow the standard reset/step interface,
similar to Gymnasium but intentionally lighter-weight to avoid the dependency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class EnvStepResult:
    """Result of a single environment step."""

    observation: Image.Image  # PIL image to feed to the VLM
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """
    Abstract base environment for VLA appendage training.

    Subclasses must implement:
      - reset()        → initial observation
      - step(action)   → EnvStepResult
      - expert_action() → optimal action for behavioral cloning
      - prompt property → text prompt for the VLM
      - image_size property → (width, height)
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> Image.Image:
        """
        Reset environment to a new episode.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            First visual observation.
        """
        ...

    @abstractmethod
    def step(self, action: Any) -> EnvStepResult:
        """
        Apply an action and advance the environment.

        Args:
            action: Action produced by an appendage.decode() or raw tensor.

        Returns:
            EnvStepResult with (observation, reward, done, info).
        """
        ...

    @abstractmethod
    def expert_action(self) -> Any:
        """
        Return the optimal action for the current state.

        Used to generate demonstrations for behavioral cloning.
        Should be deterministic given the current state.
        """
        ...

    @property
    @abstractmethod
    def prompt(self) -> str:
        """Natural-language prompt given to the VLM alongside the image."""
        ...

    @property
    @abstractmethod
    def image_size(self) -> tuple[int, int]:
        """(width, height) in pixels of rendered observations."""
        ...

    @property
    def max_steps(self) -> int:
        """Maximum steps per episode (override in subclasses)."""
        return 100
