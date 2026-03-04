"""
Training loop implementations for VLA appendage grafts.

Two training phases:

  BCTrainer  — Behavioral Cloning (supervised, from expert demonstrations)
               Fast convergence; teaches the head to read the VLM's features.
               Should always run first.

  RLTrainer  — REINFORCE policy gradient (environment reward signal)
               Fine-tunes the policy using actual episode rewards.
               Runs after BC has given a reasonable starting point.

Both trainers are environment-agnostic. The environment provides:
  - reset() → PIL Image
  - step(action) → EnvStepResult
  - expert_action() → action (for BC)
  - prompt → str

Both trainers support any action appendage (joystick, dpad, etc.) through the
polymorphic action_loss() and decode() interfaces.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..appendages.button import ButtonAppendage, MultiButtonAppendage
from ..appendages.dpad import DPadAppendage
from ..appendages.joystick import JoystickAppendage
from ..appendages.touchscreen import TouchscreenAppendage
from ..environments.base import BaseEnvironment
from ..grafting.freezing import DEFAULT_CURRICULUM, FreezingCurriculum, FreezingStage
from ..grafting.graft import VLAGraft


@dataclass
class TrainerConfig:
    """Shared configuration for BC and RL trainers."""

    # ── Optimizer ────────────────────────────────────────────────────────
    appendage_lr: float = 1e-4      # Learning rate for the action head
    vlm_lr: float = 1e-6            # LR for unfrozen VLM layers (much smaller)
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ── BC ───────────────────────────────────────────────────────────────
    bc_steps: int = 1000
    # Mini-batch: number of (image, expert_action) pairs per gradient update
    batch_size: int = 4
    # Random env steps taken before sampling an expert state (diversifies BC data)
    bc_warmup_steps_range: tuple[int, int] = (0, 8)

    # ── RL ───────────────────────────────────────────────────────────────
    rl_steps: int = 500
    rl_max_steps_per_episode: int = 30  # cap per-episode length during RL (avoids very long rollouts)
    rl_gamma: float = 0.99
    rl_entropy_coef: float = 0.02   # Entropy bonus for discrete actions
    rl_explore_noise: float = 0.10  # Gaussian noise scale for continuous actions
    rl_episodes_per_update: int = 4  # Episodes to collect before one RL update

    # ── Logging & checkpointing ──────────────────────────────────────────
    log_every: int = 50
    eval_every: int = 200
    eval_episodes: int = 5
    save_every: int = 500
    save_dir: str = "model_checkpoints"

    # ── Curriculum ───────────────────────────────────────────────────────
    freezing_stages: list[FreezingStage] = field(
        default_factory=lambda: DEFAULT_CURRICULUM
    )


# ── Shared helpers ────────────────────────────────────────────────────────────

def _preprocess(processor, image, prompt, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Convert a PIL Image + prompt to model-ready tensors.

    Handles two processor families:
      - Chat-template processors (SmolVLM / Idefics3, LLaVA-Next, etc.):
        require the text to embed image tokens via apply_chat_template().
      - Legacy processors (BLIP, older LLaVA): accept plain text + image.
    """
    if hasattr(processor, "apply_chat_template"):
        # Chat-template path: image token injected automatically
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            images=[image],
            text=[text],
            return_tensors="pt",
            padding=True,
        )
    else:
        # Legacy path
        try:
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            inputs = processor(image, prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _to_action_tensor(
    appendage: nn.Module,
    expert_actions: list[Any],
    device: torch.device,
) -> torch.Tensor:
    """Convert a list of expert actions to a target tensor matching the appendage type."""
    if isinstance(appendage, (JoystickAppendage, TouchscreenAppendage)):
        targets = []
        for a in expert_actions:
            if hasattr(a, "x"):
                targets.append([a.x, a.y])
            elif isinstance(a, (tuple, list)):
                targets.append([float(a[0]), float(a[1])])
            else:
                targets.append(list(a))
        return torch.tensor(targets, dtype=torch.float32, device=device)

    elif isinstance(appendage, DPadAppendage):
        targets = []
        for a in expert_actions:
            if hasattr(a, "value"):
                targets.append(a.value)
            else:
                targets.append(int(a))
        return torch.tensor(targets, dtype=torch.long, device=device)

    elif isinstance(appendage, ButtonAppendage):
        targets = [float(a) for a in expert_actions]
        return torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(-1)

    elif isinstance(appendage, MultiButtonAppendage):
        targets = []
        for a in expert_actions:
            if isinstance(a, (list, tuple)):
                targets.append([float(v) for v in a])
            else:
                targets.append(list(a))
        return torch.tensor(targets, dtype=torch.float32, device=device)

    else:
        raise NotImplementedError(f"Unknown appendage type: {type(appendage).__name__}")


def _select_action(appendage: nn.Module, action_out: torch.Tensor, explore: bool = False):
    """
    Select a concrete action from the appendage output.

    For inference (explore=False): argmax / raw values.
    For exploration (explore=True): sample / add noise.
    """
    if isinstance(appendage, DPadAppendage):
        if explore:
            return int(appendage.sample(action_out, temperature=1.0).item())
        return int(appendage.argmax(action_out).item())
    elif isinstance(appendage, ButtonAppendage):
        val = float(action_out.squeeze().item())
        return val
    elif isinstance(appendage, MultiButtonAppendage):
        return action_out.squeeze(0).detach().cpu().tolist()
    else:
        # JoystickAppendage, TouchscreenAppendage, and any future continuous heads
        raw = action_out.squeeze(0).detach().cpu().tolist()
        return raw


# ── BCTrainer ─────────────────────────────────────────────────────────────────

class BCTrainer:
    """
    Behavioral Cloning trainer.

    Collects (observation, expert_action) pairs by rolling out the environment's
    expert policy, then trains the action head to match via supervised loss.

    The VLM backbone is kept frozen initially; the FreezingCurriculum gradually
    unfreezes layers as training progresses.
    """

    def __init__(
        self,
        graft: VLAGraft,
        processor,
        environment: BaseEnvironment,
        config: TrainerConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.graft = graft
        self.processor = processor
        self.env = environment
        self.config = config or TrainerConfig()
        self.device = torch.device(device)
        self._global_step = 0
        self._metrics: list[dict] = []

        self.graft.to(self.device)

        self.curriculum = FreezingCurriculum(
            vlm=self.graft.vlm,
            stages=self.config.freezing_stages,
            on_stage_change=lambda s: self._on_stage_change(s),
        )
        self._optimizer = self._build_optimizer()

    def _build_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(
            self.graft.parameter_groups(
                appendage_lr=self.config.appendage_lr,
                vlm_lr=self.config.vlm_lr,
            ),
            weight_decay=self.config.weight_decay,
        )

    def _on_stage_change(self, stage: FreezingStage):
        # Rebuild optimizer so newly unfrozen VLM params are included
        self._optimizer = self._build_optimizer()
        print(f"  [BCTrainer] Optimizer rebuilt for stage: {stage.name}")

    # ------------------------------------------------------------------ #
    #  Data collection                                                     #
    # ------------------------------------------------------------------ #

    def _collect_bc_batch(self) -> tuple[list, list]:
        """
        Collect a batch of (observation, expert_action) pairs.

        Each sample comes from a different random state in the environment —
        achieved by resetting and then taking a random number of expert steps.
        This produces a diverse distribution of training states.
        """
        images, expert_actions = [], []
        lo, hi = self.config.bc_warmup_steps_range

        for _ in range(self.config.batch_size):
            obs = self.env.reset()
            n_random = random.randint(lo, hi)
            for _ in range(n_random):
                act = self.env.expert_action()
                result = self.env.step(act)
                obs = result.observation
                if result.done:
                    obs = self.env.reset()
                    break
            images.append(obs)
            expert_actions.append(self.env.expert_action())

        return images, expert_actions

    # ------------------------------------------------------------------ #
    #  Training step                                                       #
    # ------------------------------------------------------------------ #

    def train_step(self) -> dict:
        """One BC gradient update step."""
        self.graft.train()

        # Update curriculum (may rebuild optimizer via callback)
        self.curriculum.step(self._global_step)

        images, expert_actions = self._collect_bc_batch()
        self._optimizer.zero_grad()
        total_loss = 0.0

        for img, expert_act in zip(images, expert_actions):
            inputs = _preprocess(self.processor, img, self.env.prompt, self.device)
            out = self.graft(**inputs)
            pred = out["action"]  # [1, *action_shape]

            target = _to_action_tensor(self.graft.appendage, [expert_act], self.device)
            loss = self.graft.appendage.action_loss(pred, target)
            loss = loss / self.config.batch_size
            loss.backward()
            total_loss += float(loss)

        # Gradient clipping across all trainable parameters
        trainable = list(self.graft.appendage.parameters()) + [
            p for p in self.graft.vlm.parameters() if p.requires_grad
        ]
        nn.utils.clip_grad_norm_(trainable, self.config.grad_clip)
        self._optimizer.step()

        self._global_step += 1
        return {
            "step": self._global_step,
            "bc/loss": total_loss,
            "stage": (
                self.curriculum.current_stage.name
                if self.curriculum.current_stage
                else "none"
            ),
        }

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, n_episodes: int | None = None) -> dict:
        n_episodes = n_episodes or self.config.eval_episodes
        self.graft.eval()
        successes = 0
        rewards = []

        for ep_seed in range(n_episodes):
            obs = self.env.reset(seed=ep_seed * 7)
            ep_reward = 0.0
            max_steps = self.env.max_steps

            for _ in range(max_steps):
                with torch.no_grad():
                    inputs = _preprocess(
                        self.processor, obs, self.env.prompt, self.device
                    )
                    out = self.graft(**inputs)
                action_val = _select_action(self.graft.appendage, out["action"])
                result = self.env.step(action_val)
                ep_reward += result.reward
                obs = result.observation
                if result.done:
                    if result.info.get("success"):
                        successes += 1
                    break

            rewards.append(ep_reward)

        return {
            "eval/success_rate": successes / n_episodes,
            "eval/mean_reward": sum(rewards) / len(rewards),
        }

    # ------------------------------------------------------------------ #
    #  Full training loop                                                  #
    # ------------------------------------------------------------------ #

    def train(self) -> list[dict]:
        print(f"{'='*60}")
        print(f"  BC Training — {type(self.graft.appendage).__name__}")
        print(f"  Steps: {self.config.bc_steps}  |  Device: {self.device}")
        print(f"  Environment: {type(self.env).__name__}")
        print(f"{'='*60}")
        start = time.time()

        for step in tqdm(range(self.config.bc_steps), desc="BC"):
            metrics = self.train_step()

            if step % self.config.log_every == 0:
                elapsed = time.time() - start
                print(
                    f"  step {step:5d}  loss={metrics['bc/loss']:.4f}  "
                    f"stage={metrics['stage']}  t={elapsed:.0f}s"
                )

            if step % self.config.eval_every == 0 and step > 0:
                ev = self.evaluate()
                print(
                    f"  → eval @ {step}: "
                    f"success={ev['eval/success_rate']:.0%}  "
                    f"reward={ev['eval/mean_reward']:.2f}"
                )
                metrics.update(ev)

            if step % self.config.save_every == 0 and step > 0:
                self.graft.save(Path(self.config.save_dir) / f"bc_step_{step:06d}")

            self._metrics.append(metrics)

        # Final eval & save
        print(f"\n  Running final BC evaluation ({self.config.eval_episodes} episodes)...")
        ev = self.evaluate(n_episodes=self.config.eval_episodes)
        print(f"  Final eval: {ev}")
        self.graft.save(Path(self.config.save_dir) / "bc_final")

        return self._metrics


# ── RLTrainer ─────────────────────────────────────────────────────────────────

class RLTrainer:
    """
    REINFORCE policy gradient trainer.

    Runs complete episodes, collects (log_prob, reward) sequences, computes
    discounted returns, and updates the policy with the policy gradient theorem.

    Best used AFTER BCTrainer has warm-started the action head, so the initial
    policy is already reasonable (important for REINFORCE's high variance).
    """

    def __init__(
        self,
        graft: VLAGraft,
        processor,
        environment: BaseEnvironment,
        config: TrainerConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.graft = graft
        self.processor = processor
        self.env = environment
        self.config = config or TrainerConfig()
        self.device = torch.device(device)
        self._global_step = 0
        self._metrics: list[dict] = []
        self.graft.to(self.device)

    # ------------------------------------------------------------------ #
    #  Episode rollout                                                     #
    # ------------------------------------------------------------------ #

    def _run_episode(self) -> tuple[list[torch.Tensor], list[float], dict]:
        """Collect one episode, returning (log_probs, rewards, final_info)."""
        obs = self.env.reset()
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        max_steps = min(self.env.max_steps, self.config.rl_max_steps_per_episode)

        self.graft.train()

        for _ in range(max_steps):
            inputs = _preprocess(self.processor, obs, self.env.prompt, self.device)
            out = self.graft(**inputs)
            action_out = out["action"]

            if isinstance(self.graft.appendage, DPadAppendage):
                probs = torch.softmax(action_out, dim=-1)
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample()
                log_probs.append(dist.log_prob(idx))
                action_val = int(idx.item())

            else:
                # Continuous: Gaussian exploration
                sigma = self.config.rl_explore_noise
                noise = torch.randn_like(action_out) * sigma
                noisy = (action_out + noise).clamp(-1.0, 1.0)
                # Log prob of the noise under N(0, sigma)
                log_prob = (
                    -0.5 * ((noise / sigma) ** 2).sum(-1)
                    - noise.shape[-1] * 0.5 * torch.log(torch.tensor(2 * 3.14159265))
                )
                log_probs.append(log_prob.squeeze())
                action_val = noisy.squeeze(0).detach().cpu().tolist()

            result = self.env.step(action_val)
            rewards.append(result.reward)
            obs = result.observation
            if result.done:
                return log_probs, rewards, result.info

        return log_probs, rewards, {}

    def _compute_returns(self, rewards: list[float]) -> torch.Tensor:
        """Discounted return G_t = Σ_{k≥0} γ^k r_{t+k}, normalized."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.config.rl_gamma * G
            returns.insert(0, G)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if ret.std() > 1e-6:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        return ret

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #

    def train(self, optimizer: optim.Optimizer | None = None) -> list[dict]:
        print(f"{'='*60}")
        print(f"  RL Fine-tuning — {type(self.graft.appendage).__name__}")
        print(f"  Steps: {self.config.rl_steps}  |  Device: {self.device}")
        print(f"{'='*60}")

        if optimizer is None:
            optimizer = optim.Adam(
                self.graft.appendage.parameters(),
                lr=self.config.appendage_lr * 0.1,
            )

        acc_log_probs: list[torch.Tensor] = []
        acc_returns: list[torch.Tensor] = []
        n_episodes = 0

        pbar = tqdm(total=self.config.rl_steps, desc="RL")
        while self._global_step < self.config.rl_steps:
            log_probs, rewards, info = self._run_episode()
            returns = self._compute_returns(rewards)

            acc_log_probs.extend(log_probs)
            acc_returns.extend(returns)
            n_episodes += 1

            if n_episodes % self.config.rl_episodes_per_update == 0:
                optimizer.zero_grad()

                lp = torch.stack(acc_log_probs)
                ret = torch.stack(acc_returns)

                # Entropy bonus for discrete actions
                policy_loss = -(lp * ret).mean()
                if isinstance(self.graft.appendage, DPadAppendage):
                    # Entropy of the last step's distribution as regularizer
                    with torch.no_grad():
                        last_inputs = _preprocess(
                            self.processor,
                            self.env.reset(),
                            self.env.prompt,
                            self.device,
                        )
                        last_out = self.graft(**last_inputs)
                    probs = torch.softmax(last_out["action"], dim=-1)
                    entropy = -(probs * probs.log()).sum(-1).mean()
                    policy_loss = policy_loss - self.config.rl_entropy_coef * entropy

                policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.graft.appendage.parameters(), self.config.grad_clip
                )
                optimizer.step()

                ep_reward = sum(rewards)
                metrics = {
                    "step": self._global_step,
                    "rl/loss": float(policy_loss),
                    "rl/episode_reward": ep_reward,
                    "rl/success": bool(info.get("success", False)),
                    "rl/n_episodes": n_episodes,
                }
                self._metrics.append(metrics)

                pbar.update(1)
                pbar.set_postfix(loss=f"{float(policy_loss):.4f}", reward=f"{ep_reward:.2f}")

                if self._global_step % self.config.log_every == 0:
                    print(
                        f"  step {self._global_step:5d}  "
                        f"loss={float(policy_loss):.4f}  "
                        f"reward={ep_reward:.2f}  "
                        f"success={info.get('success', False)}"
                    )

                acc_log_probs.clear()
                acc_returns.clear()
                self._global_step += 1

        pbar.close()
        self.graft.save(Path(self.config.save_dir) / "rl_final")
        return self._metrics
