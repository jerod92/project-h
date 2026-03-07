"""
Benchmarking suite for VLA grafts.

Provides standardized evaluation across multiple seeds, environments, and grafts.
All benchmarks run in evaluation mode (no gradients, greedy actions).

Key metrics:
  - success_rate  — fraction of episodes reaching the goal
  - mean_reward   — average total episode reward
  - std_reward    — reward standard deviation
  - mean_steps    — average steps to termination
  - median_steps  — median steps (robust to outliers)

Use compare_grafts() to run ablations: e.g. compare a fully-trained graft
against a randomly-initialized one, or compare joystick vs d-pad on the same task.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch
from PIL import Image
from tqdm import tqdm

from ..appendages.dpad import DPadAppendage
from ..appendages.joystick import JoystickAppendage
from ..environments.base import BaseEnvironment, EnvStepResult
from ..grafting.graft import VLAGraft
from ..training.trainer import _preprocess, _select_action


@dataclass
class EpisodeResult:
    """Metrics from a single evaluation episode."""

    success: bool
    total_reward: float
    n_steps: int
    final_info: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results from running N episodes on one environment."""

    env_name: str
    appendage_type: str
    n_episodes: int
    success_rate: float
    mean_reward: float
    std_reward: float
    median_reward: float
    mean_steps: float
    median_steps: float
    episodes: list[EpisodeResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        bar = "─" * 52
        return (
            f"\n{bar}\n"
            f"  Benchmark: {self.env_name} / {self.appendage_type}\n"
            f"{bar}\n"
            f"  Episodes      : {self.n_episodes}\n"
            f"  Success rate  : {self.success_rate:.1%}\n"
            f"  Reward  mean  : {self.mean_reward:+.2f}  ±{self.std_reward:.2f}\n"
            f"  Reward  median: {self.median_reward:+.2f}\n"
            f"  Steps   mean  : {self.mean_steps:.1f}\n"
            f"  Steps   median: {self.median_steps:.1f}\n"
            f"  Wall time     : {self.elapsed_seconds:.1f}s\n"
            f"{bar}"
        )

    def as_dict(self) -> dict:
        return {
            "env_name": self.env_name,
            "appendage_type": self.appendage_type,
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "median_reward": self.median_reward,
            "mean_steps": self.mean_steps,
            "median_steps": self.median_steps,
            "elapsed_seconds": self.elapsed_seconds,
        }


class BenchmarkSuite:
    """
    Runs standardized evaluation episodes for one or more environments.

    Usage::

        suite = BenchmarkSuite(graft, processor, [TargetNavEnvironment()], device="cuda")
        results = suite.run_all(n_episodes=20)
        for r in results:
            print(r)
    """

    def __init__(
        self,
        graft: VLAGraft,
        processor,
        environments: list[BaseEnvironment],
        device: str | torch.device = "cpu",
    ):
        self.graft = graft
        self.processor = processor
        self.environments = environments
        self.device = torch.device(device)
        self.graft.to(self.device)
        self.graft.eval()

    # ------------------------------------------------------------------ #
    #  Single episode                                                      #
    # ------------------------------------------------------------------ #

    def run_episode(self, env: BaseEnvironment, seed: int | None = None) -> EpisodeResult:
        obs = env.reset(seed=seed)
        total_reward = 0.0
        n_steps = 0
        max_steps = env.max_steps

        with torch.no_grad():
            for _ in range(max_steps):
                inputs = _preprocess(self.processor, obs, env.prompt, self.device)
                out = self.graft(**inputs)
                action_val = _select_action(self.graft.appendage, out["action"], explore=False)

                result = env.step(action_val)
                total_reward += result.reward
                n_steps += 1
                obs = result.observation

                if result.done:
                    return EpisodeResult(
                        success=bool(result.info.get("success", False)),
                        total_reward=total_reward,
                        n_steps=n_steps,
                        final_info=result.info,
                    )

        return EpisodeResult(
            success=False,
            total_reward=total_reward,
            n_steps=n_steps,
            final_info={},
        )

    # ------------------------------------------------------------------ #
    #  Environment benchmark                                               #
    # ------------------------------------------------------------------ #

    def run_benchmark(
        self, env: BaseEnvironment, n_episodes: int = 20, seed_offset: int = 0
    ) -> BenchmarkResult:
        """Run n_episodes on a single environment and return aggregated metrics."""
        episodes: list[EpisodeResult] = []
        t0 = time.time()

        for i in tqdm(range(n_episodes), desc=f"  {type(env).__name__}", leave=False):
            ep = self.run_episode(env, seed=seed_offset + i * 13)
            episodes.append(ep)

        rewards = [e.total_reward for e in episodes]
        steps = [e.n_steps for e in episodes]

        return BenchmarkResult(
            env_name=type(env).__name__,
            appendage_type=type(self.graft.appendage).__name__,
            n_episodes=n_episodes,
            success_rate=sum(e.success for e in episodes) / n_episodes,
            mean_reward=statistics.mean(rewards),
            std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            median_reward=statistics.median(rewards),
            mean_steps=statistics.mean(steps),
            median_steps=statistics.median(steps),
            episodes=episodes,
            elapsed_seconds=time.time() - t0,
        )

    def run_all(self, n_episodes: int = 20) -> list[BenchmarkResult]:
        """Run benchmarks across all registered environments."""
        results = []
        for env in self.environments:
            result = self.run_benchmark(env, n_episodes=n_episodes)
            results.append(result)
            print(result)
        return results


# ── Multi-graft comparison ────────────────────────────────────────────────────

def compare_grafts(
    grafts: dict[str, VLAGraft],
    processors: dict[str, object],
    environment: BaseEnvironment,
    n_episodes: int = 20,
    device: str = "cpu",
) -> dict[str, BenchmarkResult]:
    """
    Run the same benchmark for multiple grafts and print a comparison table.

    Args:
        grafts: Mapping from label → VLAGraft.
        processors: Mapping from label → HuggingFace processor.
        environment: The environment to evaluate on.
        n_episodes: Episodes per graft.
        device: Compute device.

    Returns:
        Dict mapping label → BenchmarkResult.

    Example::

        results = compare_grafts(
            grafts={"trained": trained_graft, "random": random_graft},
            processors={"trained": proc, "random": proc},
            environment=TargetNavEnvironment(),
            n_episodes=20,
        )
    """
    results: dict[str, BenchmarkResult] = {}
    for label, graft in grafts.items():
        print(f"\n[{label}]")
        suite = BenchmarkSuite(
            graft=graft,
            processor=processors[label],
            environments=[environment],
            device=device,
        )
        results[label] = suite.run_benchmark(environment, n_episodes=n_episodes)
        print(results[label])

    # Summary table
    print("\n" + "=" * 52)
    print(f"  {'Label':<20} {'Success':>8} {'Reward':>10}")
    print("  " + "─" * 40)
    for label, r in results.items():
        print(f"  {label:<20} {r.success_rate:>7.1%} {r.mean_reward:>10.2f}")
    print("=" * 52)

    return results


# ── Expert baseline ───────────────────────────────────────────────────────────

def run_expert_baseline(
    environment: BaseEnvironment,
    n_episodes: int = 20,
    seed_offset: int = 0,
) -> BenchmarkResult:
    """
    Evaluate the environment's built-in expert policy.

    Useful for establishing an upper bound on achievable performance.
    """
    episodes: list[EpisodeResult] = []
    t0 = time.time()

    for i in range(n_episodes):
        obs = environment.reset(seed=seed_offset + i * 13)
        total_reward = 0.0
        n_steps = 0

        for _ in range(environment.max_steps):
            action = environment.expert_action()
            result = environment.step(action)
            total_reward += result.reward
            n_steps += 1
            if result.done:
                episodes.append(
                    EpisodeResult(
                        success=bool(result.info.get("success", False)),
                        total_reward=total_reward,
                        n_steps=n_steps,
                        final_info=result.info,
                    )
                )
                break
        else:
            episodes.append(
                EpisodeResult(
                    success=False,
                    total_reward=total_reward,
                    n_steps=n_steps,
                )
            )

    rewards = [e.total_reward for e in episodes]
    steps = [e.n_steps for e in episodes]

    result = BenchmarkResult(
        env_name=type(environment).__name__,
        appendage_type="ExpertPolicy",
        n_episodes=n_episodes,
        success_rate=sum(e.success for e in episodes) / n_episodes,
        mean_reward=statistics.mean(rewards),
        std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        median_reward=statistics.median(rewards),
        mean_steps=statistics.mean(steps),
        median_steps=statistics.median(steps),
        episodes=episodes,
        elapsed_seconds=time.time() - t0,
    )
    print("[Expert baseline]", result)
    return result
