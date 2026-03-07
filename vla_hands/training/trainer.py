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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
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

    # ── RL (PPO) ─────────────────────────────────────────────────────────
    rl_steps: int = 500
    rl_max_steps_per_episode: int = 30  # cap per-episode rollout length
    rl_gamma: float = 0.99              # discount factor
    rl_gae_lambda: float = 0.95         # GAE smoothing parameter (λ)
    rl_ppo_clip: float = 0.2            # PPO surrogate clip ratio (ε)
    rl_ppo_epochs: int = 4              # gradient epochs per rollout batch
    rl_entropy_coef: float = 0.02       # entropy bonus (all action types)
    rl_value_coef: float = 0.5          # value function loss weight
    rl_action_std: float = 0.3          # std for continuous action distributions
    rl_episodes_per_update: int = 4     # episodes per rollout batch

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


def _preprocess_batch(
    processor, images: list, prompt: str, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Batch-preprocess multiple PIL Images + a shared prompt.

    Processes all images in a single call, producing properly padded batch
    tensors.  Falls back to individual processing + concatenation if the
    processor doesn't support batched inputs.
    """
    n = len(images)
    if hasattr(processor, "apply_chat_template"):
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
            images=images,
            text=[text] * n,
            return_tensors="pt",
            padding=True,
        )
    else:
        try:
            inputs = processor(
                images=images,
                text=[prompt] * n,
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            # Fallback: process individually and stack
            batch = [_preprocess(processor, img, prompt, device) for img in images]
            keys = batch[0].keys()
            return {k: torch.cat([b[k] for b in batch], dim=0) for k in keys}
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

    def _get_autocast_kwargs(self) -> dict:
        enabled = self.device.type in ("cuda", "mps")
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device.type == "mps":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        return {"device_type": self.device.type, "dtype": dtype, "enabled": enabled}

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

        # Single batched VLM forward pass instead of one-at-a-time
        inputs = _preprocess_batch(
            self.processor, images, self.env.prompt, self.device
        )
        out = self.graft(**inputs)
        pred = out["action"]  # [batch, *action_shape]

        target = _to_action_tensor(self.graft.appendage, expert_actions, self.device)
        loss = self.graft.appendage.action_loss(pred, target)
        loss.backward()

        # Gradient clipping across all trainable parameters
        trainable = list(self.graft.appendage.parameters()) + [
            p for p in self.graft.vlm.parameters() if p.requires_grad
        ]
        nn.utils.clip_grad_norm_(trainable, self.config.grad_clip)
        self._optimizer.step()

        self._global_step += 1
        return {
            "step": self._global_step,
            "bc/loss": float(loss),
            "stage": (
                self.curriculum.current_stage.name
                if self.curriculum.current_stage
                else "none"
            ),
        }

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
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
                inputs = _preprocess(
                    self.processor, obs, self.env.prompt, self.device
                )
                
                with torch.autocast(**self._get_autocast_kwargs()):
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

@dataclass
class _RolloutStep:
    """One environment transition collected during a PPO rollout."""
    features: torch.Tensor   # VLM action_features, cached + detached, shape [hidden_dim]
    u: torch.Tensor          # pre-squash sample (continuous) or sampled idx (discrete)
    log_prob: torch.Tensor   # log π_old(a|s), scalar
    entropy: torch.Tensor    # H[π(·|s)], scalar
    value: torch.Tensor      # V̂(s) from value head, scalar
    squash: str              # "tanh" | "sigmoid" | "categorical"


class RLTrainer:
    """
    PPO (Proximal Policy Optimization) trainer for VLA action heads.

    Key improvements over vanilla REINFORCE:
    - Clipped surrogate objective prevents large destructive policy updates.
    - Learned value head (tiny 2-layer MLP on cached VLM features) provides a
      proper baseline, reducing gradient variance dramatically.
    - GAE(λ) advantage estimation smoothly blends bias and variance trade-offs.
    - Multiple PPO epochs per rollout batch amortise expensive VLM forward passes
      (only the small action head is re-run in epochs 2+).
    - Proper squashing-corrected log-probs for all action types:
        - DPad         → Categorical distribution
        - Joystick     → Normal in pre-tanh space (SAC-style)
        - Touchscreen / Button / MultiButton → Normal in pre-sigmoid (logit) space

    Best used AFTER BCTrainer has warm-started the action head.
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

        # Small value head — re-uses cached VLM features, so it's cheap.
        hidden_dim = self.graft.config.hidden_dim
        self._value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        ).to(self.device)

    def _get_autocast_kwargs(self) -> dict:
        enabled = self.device.type in ("cuda", "mps")
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device.type == "mps":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        return {"device_type": self.device.type, "dtype": dtype, "enabled": enabled}

    # ------------------------------------------------------------------ #
    #  Squashing helpers                                                   #
    # ------------------------------------------------------------------ #

    def _squash_type(self) -> str:
        """Determine the action squashing used by the current appendage."""
        app = self.graft.appendage
        if isinstance(app, DPadAppendage):
            return "categorical"
        elif isinstance(app, JoystickAppendage):
            return "tanh"   # JoystickAppendage ends with nn.Tanh → output in [-1, 1]
        else:
            return "sigmoid"  # Button / MultiButton / Touchscreen → sigmoid → [0, 1]

    @staticmethod
    def _log_prob_and_entropy(
        action_out: torch.Tensor,
        sigma: float,
        squash: str,
        u_sample: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log-prob, entropy, and pre-squash sample for the policy distribution.

        Args:
            action_out: network output [batch, *] (already squashed for continuous).
            sigma:      std for continuous Normal distribution.
            squash:     one of "categorical", "tanh", "sigmoid".
            u_sample:   if provided, evaluate log-prob at this pre-squash value
                        instead of drawing a fresh sample (used in PPO re-evaluation).

        Returns:
            (u, log_prob, entropy) — all scalars / [batch] tensors.
        """
        if squash == "categorical":
            dist = Categorical(logits=action_out.squeeze(0))
            u = dist.sample() if u_sample is None else u_sample
            return u, dist.log_prob(u), dist.entropy()

        if squash == "tanh":
            # For continuous joystick actions, don't use atanh inverse on clipped actions.
            # action_out is already the mean in the squashed space, but PPO needs unconstrained.
            # To fix gradient saturation, we assume action_out is un-squashed logits if we're in RL.
            # Actually, to make this work seamlessly without changing JoystickAppendage, we will
            # use a looser clamp to prevent complete gradient annihilation, and ensure the gradient flows.
            u_mean = torch.atanh(action_out.clamp(-1 + 1e-4, 1 - 1e-4))
            dist = Normal(u_mean, torch.full_like(u_mean, sigma))
            u = dist.rsample() if u_sample is None else u_sample
            action = torch.tanh(u)
            # Jacobian correction for tanh squashing (SAC-style)
            lp = dist.log_prob(u).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            ent = dist.entropy().sum(-1)
        else:  # sigmoid
            # action_out ∈ [0, 1]; invert sigmoid (logit) to get unbounded mean
            u_mean = torch.logit(action_out.clamp(1e-4, 1 - 1e-4))
            dist = Normal(u_mean, torch.full_like(u_mean, sigma))
            u = dist.rsample() if u_sample is None else u_sample
            action = torch.sigmoid(u)
            # Jacobian correction for sigmoid squashing
            lp = dist.log_prob(u).sum(-1) - torch.log(action * (1 - action) + 1e-6).sum(-1)
            ent = dist.entropy().sum(-1)

        return u, lp.squeeze(), ent.squeeze()

    # ------------------------------------------------------------------ #
    #  Episode rollout                                                     #
    # ------------------------------------------------------------------ #

    def _run_episode(self) -> tuple[list[_RolloutStep], list[float], dict]:
        """
        Collect one episode.

        Returns:
            steps:   list of _RolloutStep (one per env step taken)
            rewards: per-step rewards
            info:    final env info dict
        """
        obs = self.env.reset()
        steps: list[_RolloutStep] = []
        rewards: list[float] = []
        max_steps = min(self.env.max_steps, self.config.rl_max_steps_per_episode)
        squash = self._squash_type()
        sigma = self.config.rl_action_std

        self.graft.train()
        self._value_head.train()

        for _ in range(max_steps):
            inputs = _preprocess(self.processor, obs, self.env.prompt, self.device)
            
            with torch.autocast(**self._get_autocast_kwargs()):
                out = self.graft(**inputs)
            
            action_out = out["action"]
            # Cache features detached from the VLM graph — reused for PPO epochs
            features = out["action_features"].squeeze(0).detach()  # [hidden_dim]
            value = self._value_head(features.unsqueeze(0)).squeeze()  # scalar

            u, log_prob, entropy = self._log_prob_and_entropy(action_out, sigma, squash)

            # Convert sample to concrete environment action
            if squash == "categorical":
                action_val = int(u.item())
            elif squash == "tanh":
                action_val = torch.tanh(u).squeeze(0).detach().cpu().tolist()
                u = u.squeeze(0).detach()  # [D]
            else:
                action_val = torch.sigmoid(u).squeeze(0).detach().cpu().tolist()
                u = u.squeeze(0).detach()  # [D]

            steps.append(_RolloutStep(
                features=features,
                u=u.detach() if isinstance(u, torch.Tensor) else u,
                log_prob=log_prob,
                entropy=entropy,
                value=value,
                squash=squash,
            ))

            result = self.env.step(action_val)
            rewards.append(result.reward)
            obs = result.observation
            if result.done:
                return steps, rewards, result.info

        return steps, rewards, {}

    # ------------------------------------------------------------------ #
    #  GAE advantage estimation                                            #
    # ------------------------------------------------------------------ #

    def _compute_gae(
        self, rewards: list[float], values: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generalised Advantage Estimation (GAE-λ).

        A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + …
        where δ_t = r_t + γ V(s_{t+1}) − V(s_t).

        Returns:
            advantages: [T] (un-normalised; caller normalises across the batch)
            returns:    [T] discounted return targets for the value head
        """
        T = len(rewards)
        vals = torch.stack(values).detach()   # [T]
        adv = torch.zeros(T, device=self.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_val = vals[t + 1].item() if t + 1 < T else 0.0
            delta = rewards[t] + self.config.rl_gamma * next_val - vals[t].item()
            last_gae = delta + self.config.rl_gamma * self.config.rl_gae_lambda * last_gae
            adv[t] = last_gae

        returns = adv + vals   # V + A = discounted return (value target)
        return adv, returns

    # ------------------------------------------------------------------ #
    #  PPO update                                                          #
    # ------------------------------------------------------------------ #

    def _ppo_update(
        self,
        all_steps: list[_RolloutStep],
        all_adv: torch.Tensor,   # [T], batch-normalised
        all_ret: torch.Tensor,   # [T], value targets
        optimizer: optim.Optimizer,
    ) -> dict:
        """
        K epochs of PPO mini-batch updates.

        Only re-runs the small action head + value head using cached VLM features —
        no expensive VLM forward passes in epochs 2+.
        """
        squash = all_steps[0].squash
        sigma = self.config.rl_action_std

        features = torch.stack([s.features for s in all_steps])   # [T, hidden_dim]
        old_lp = torch.stack([s.log_prob for s in all_steps]).detach()  # [T]
        old_u = torch.stack([s.u for s in all_steps])             # [T] or [T, D]
        clip_eps = self.config.rl_ppo_clip

        total_pg = total_vf = total_ent = 0.0

        for _ in range(self.config.rl_ppo_epochs):
            # Re-evaluate action head on cached features (cheap — no VLM)
            new_action_out = self.graft.appendage(features)       # [T, *]
            new_values = self._value_head(features).squeeze(-1)   # [T]

            _, new_lp, entropy = self._log_prob_and_entropy(
                new_action_out, sigma, squash, u_sample=old_u
            )

            # PPO clipped surrogate objective
            ratio = torch.exp(new_lp - old_lp)
            pg_loss = torch.max(
                -ratio * all_adv,
                -ratio.clamp(1 - clip_eps, 1 + clip_eps) * all_adv,
            ).mean()

            # Value function loss (MSE to GAE-estimated returns)
            vf_loss = F.mse_loss(new_values, all_ret)

            loss = (
                pg_loss
                + self.config.rl_value_coef * vf_loss
                - self.config.rl_entropy_coef * entropy.mean()
            )

            optimizer.zero_grad()
            loss.backward()
            all_params = (
                list(self.graft.appendage.parameters())
                + list(self._value_head.parameters())
            )
            nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)
            optimizer.step()

            total_pg += pg_loss.item()
            total_vf += vf_loss.item()
            total_ent += entropy.mean().item()

        K = self.config.rl_ppo_epochs
        return {
            "rl/pg_loss": total_pg / K,
            "rl/vf_loss": total_vf / K,
            "rl/entropy": total_ent / K,
        }

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #

    def train(self, optimizer: optim.Optimizer | None = None) -> list[dict]:
        print(f"{'='*60}")
        print(f"  RL Fine-tuning (PPO) — {type(self.graft.appendage).__name__}")
        print(f"  Steps: {self.config.rl_steps}  |  Device: {self.device}")
        print(
            f"  clip={self.config.rl_ppo_clip}  epochs={self.config.rl_ppo_epochs}"
            f"  GAE λ={self.config.rl_gae_lambda}  std={self.config.rl_action_std}"
        )
        print(f"{'='*60}")

        if optimizer is None:
            optimizer = optim.Adam(
                list(self.graft.appendage.parameters())
                + list(self._value_head.parameters()),
                lr=self.config.appendage_lr * 0.1,
            )

        acc_steps: list[_RolloutStep] = []
        acc_rewards_per_ep: list[list[float]] = []
        n_episodes = 0

        pbar = tqdm(total=self.config.rl_steps, desc="RL")
        while self._global_step < self.config.rl_steps:
            steps, rewards, info = self._run_episode()
            acc_steps.extend(steps)
            acc_rewards_per_ep.append(rewards)
            n_episodes += 1

            if n_episodes % self.config.rl_episodes_per_update == 0:
                # Per-episode GAE, then concatenate across the batch
                all_adv_list, all_ret_list = [], []
                start = 0
                for ep_rewards in acc_rewards_per_ep:
                    T = len(ep_rewards)
                    ep_steps = acc_steps[start : start + T]
                    adv, ret = self._compute_gae(ep_rewards, [s.value for s in ep_steps])
                    all_adv_list.append(adv)
                    all_ret_list.append(ret)
                    start += T

                all_adv = torch.cat(all_adv_list)
                all_ret = torch.cat(all_ret_list)

                # Normalise advantages across the full batch
                if all_adv.std() > 1e-6:
                    all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

                update_metrics = self._ppo_update(acc_steps, all_adv, all_ret, optimizer)

                mean_ep_reward = (
                    sum(sum(r) for r in acc_rewards_per_ep) / len(acc_rewards_per_ep)
                )
                metrics = {
                    "step": self._global_step,
                    **update_metrics,
                    "rl/episode_reward": mean_ep_reward,
                    "rl/success": bool(info.get("success", False)),
                    "rl/n_episodes": n_episodes,
                }
                self._metrics.append(metrics)

                pbar.update(1)
                pbar.set_postfix(
                    pg=f"{update_metrics['rl/pg_loss']:.4f}",
                    vf=f"{update_metrics['rl/vf_loss']:.4f}",
                    rew=f"{mean_ep_reward:.2f}",
                )

                if self._global_step % self.config.log_every == 0:
                    print(
                        f"  step {self._global_step:5d}  "
                        f"pg={update_metrics['rl/pg_loss']:.4f}  "
                        f"vf={update_metrics['rl/vf_loss']:.4f}  "
                        f"ent={update_metrics['rl/entropy']:.4f}  "
                        f"rew={mean_ep_reward:.2f}  "
                        f"success={info.get('success', False)}"
                    )

                acc_steps.clear()
                acc_rewards_per_ep.clear()
                self._global_step += 1

        pbar.close()
        self.graft.save(Path(self.config.save_dir) / "rl_final")
        return self._metrics
