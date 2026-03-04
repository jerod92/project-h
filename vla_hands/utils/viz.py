"""
Training visualisation utilities.

Works with the metric dicts returned by BCTrainer.train() and RLTrainer.train(),
as well as the combined dict from TrainingCurriculum.run().

Usage::

    from vla_hands.utils import plot_training_curves, TrainingSummary

    metrics = curriculum.run()                  # returned by TrainingCurriculum
    plot_training_curves(metrics)               # pop-up matplotlib figure
    summary = TrainingSummary(metrics)
    print(summary)
    summary.plot()                              # same figure, returned handle
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ── TrainingSummary ───────────────────────────────────────────────────────────

@dataclass
class TrainingSummary:
    """
    Human-readable summary of a completed training run.

    Constructed from the list of metric dicts returned by a Trainer's
    .train() method.
    """
    n_bc_steps: int = 0
    n_rl_steps: int = 0
    final_bc_loss: float | None = None
    best_success_rate: float | None = None
    final_success_rate: float | None = None
    final_mean_reward: float | None = None
    best_rl_reward: float | None = None

    @classmethod
    def from_metrics(cls, metrics: list[dict[str, Any]] | dict) -> "TrainingSummary":
        """Build from a flat list of step dicts or a {"bc": [...], "rl": [...]} dict."""
        if isinstance(metrics, dict):
            bc = metrics.get("bc", [])
            rl = metrics.get("rl", [])
            all_m = bc + rl
        else:
            all_m = metrics
            bc = [m for m in all_m if "bc/loss" in m]
            rl = [m for m in all_m if "rl/loss" in m]

        obj = cls()
        obj.n_bc_steps = len(bc)
        obj.n_rl_steps = len(rl)

        bc_losses = [m["bc/loss"] for m in bc if "bc/loss" in m]
        if bc_losses:
            obj.final_bc_loss = bc_losses[-1]

        success_vals = [m["eval/success_rate"] for m in all_m if "eval/success_rate" in m]
        if success_vals:
            obj.best_success_rate = max(success_vals)
            obj.final_success_rate = success_vals[-1]

        reward_vals = [m["eval/mean_reward"] for m in all_m if "eval/mean_reward" in m]
        if reward_vals:
            obj.final_mean_reward = reward_vals[-1]

        rl_rewards = [m["rl/episode_reward"] for m in rl if "rl/episode_reward" in m]
        if rl_rewards:
            obj.best_rl_reward = max(rl_rewards)

        return obj

    def __str__(self) -> str:
        lines = [
            "┌─ Training Summary " + "─" * 42,
            f"│  BC steps:          {self.n_bc_steps}",
            f"│  RL steps:          {self.n_rl_steps}",
        ]
        if self.final_bc_loss is not None:
            lines.append(f"│  Final BC loss:     {self.final_bc_loss:.4f}")
        if self.best_success_rate is not None:
            lines.append(f"│  Best success:      {self.best_success_rate:.0%}")
        if self.final_success_rate is not None:
            lines.append(f"│  Final success:     {self.final_success_rate:.0%}")
        if self.final_mean_reward is not None:
            lines.append(f"│  Final mean reward: {self.final_mean_reward:+.2f}")
        if self.best_rl_reward is not None:
            lines.append(f"│  Best RL reward:    {self.best_rl_reward:+.2f}")
        lines.append("└" + "─" * 61)
        return "\n".join(lines)

    def plot(self, **kwargs):
        """Alias for plot_training_curves() with the metrics stored internally."""
        raise AttributeError(
            "Store the raw metrics list and call plot_training_curves(metrics) instead."
        )


# ── plot_training_curves ──────────────────────────────────────────────────────

def plot_training_curves(
    metrics: list[dict] | dict,
    title: str = "",
    figsize: tuple[int, int] | None = None,
    smooth: int = 5,
):
    """
    Plot BC loss, RL reward, and success rate from training metrics.

    Args:
        metrics:  List of step dicts from BCTrainer/RLTrainer, OR a dict
                  ``{"joystick": [...], "dpad": [...]}`` for multi-graft runs.
        title:    Optional figure title.
        figsize:  Override figure size (width, height in inches).
        smooth:   Running-average window size for smoothing curves.

    Returns:
        The matplotlib Figure object (for saving / further customisation).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")

    # Normalise input → dict of {label: [metric_dicts]}
    if isinstance(metrics, list):
        named = {"run": metrics}
    elif isinstance(metrics, dict):
        # Check if it looks like {"bc": [...], "rl": [...]}  (single-run split)
        if all(isinstance(v, list) for v in metrics.values()) and set(metrics.keys()) <= {"bc", "rl"}:
            named = {"run": metrics.get("bc", []) + metrics.get("rl", [])}
        else:
            named = metrics   # type: ignore[assignment]
    else:
        raise TypeError(f"metrics must be list or dict, got {type(metrics)}")

    # Determine subplot layout
    has_bc = any("bc/loss" in m for ms in named.values() for m in ms)
    has_rl = any("rl/episode_reward" in m for ms in named.values() for m in ms)
    has_sr = any("eval/success_rate" in m for ms in named.values() for m in ms)
    n_plots = sum([has_bc, has_rl, has_sr])
    if n_plots == 0:
        print("[viz] No plottable metrics found.")
        return None

    fig_w = figsize[0] if figsize else max(6, 4 * n_plots)
    fig_h = figsize[1] if figsize else 4
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_w, fig_h))
    if n_plots == 1:
        axes = [axes]

    ax_iter = iter(axes)
    colors = ["#4C9BE8", "#5BC46A", "#E8804C", "#C45BE8", "#E8C84C"]

    def _smooth(vals: list[float], w: int) -> list[float]:
        if w <= 1 or len(vals) < w:
            return vals
        return [
            sum(vals[max(0, i - w + 1) : i + 1]) / min(i + 1, w)
            for i in range(len(vals))
        ]

    if has_bc:
        ax = next(ax_iter)
        for i, (label, ms) in enumerate(named.items()):
            steps = [m["step"] for m in ms if "bc/loss" in m]
            vals = [m["bc/loss"] for m in ms if "bc/loss" in m]
            ax.plot(steps, _smooth(vals, smooth), label=label,
                    color=colors[i % len(colors)], linewidth=2)
        ax.set_title("BC Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        if len(named) > 1:
            ax.legend(fontsize=8)

    if has_rl:
        ax = next(ax_iter)
        for i, (label, ms) in enumerate(named.items()):
            steps = [m["step"] for m in ms if "rl/episode_reward" in m]
            vals = [m["rl/episode_reward"] for m in ms if "rl/episode_reward" in m]
            ax.plot(steps, _smooth(vals, smooth), label=label,
                    color=colors[i % len(colors)], linewidth=2)
        ax.set_title("RL Episode Reward")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(alpha=0.3)
        if len(named) > 1:
            ax.legend(fontsize=8)

    if has_sr:
        ax = next(ax_iter)
        for i, (label, ms) in enumerate(named.items()):
            steps = [m["step"] for m in ms if "eval/success_rate" in m]
            vals = [m["eval/success_rate"] * 100 for m in ms if "eval/success_rate" in m]
            ax.plot(steps, vals, "o-", label=label,
                    color=colors[i % len(colors)], linewidth=2, markersize=5)
        ax.set_title("Success Rate (%)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Success (%)")
        ax.set_ylim(-5, 105)
        ax.axhline(50, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(alpha=0.3)
        if len(named) > 1:
            ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    plt.show()
    return fig
