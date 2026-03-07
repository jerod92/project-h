"""Utility helpers: GIF recording, training visualisation, benchmarking."""

from .gif import record_expert_gif, save_rollout_gif
from .viz import plot_training_curves, TrainingSummary
from .benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    EpisodeResult,
    compare_grafts,
    run_expert_baseline,
)

__all__ = [
    # GIF
    "save_rollout_gif",
    "record_expert_gif",
    # Visualisation
    "plot_training_curves",
    "TrainingSummary",
    # Benchmarking
    "BenchmarkResult",
    "BenchmarkSuite",
    "EpisodeResult",
    "compare_grafts",
    "run_expert_baseline",
]
