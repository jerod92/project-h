"""
vla_hands.auto — Auto-configuration: factories, mappings, and one-call training.

Three main capabilities:

1. String-name factories
   ``make_env(name, **kwargs)``        → BaseEnvironment
   ``make_appendage(name, hidden_dim)`` → BaseAppendage
   ``make_graft(vlm, names, hidden_dim)`` → VLAGraft or CompositeGraft

2. Appendage-to-environment mapping
   ``recommended_envs(appendage_names)`` → list of env class names + descriptions
   ``APPENDAGE_ENV_MAP``                  → full mapping dict

3. Auto curriculum
   ``auto_curriculum(vlm, processor, appendage_names, hidden_dim, budget_steps)``
   → runs the full training pipeline for any appendage combination and returns metrics

Usage::

    from vla_hands.auto import make_env, make_appendage, auto_curriculum

    # One-call training for a joystick + button combo
    metrics = auto_curriculum(
        vlm=vlm, processor=processor,
        appendage_names=["joystick", "button"],
        hidden_dim=1152,
        budget_steps=600,
        device="cuda",
    )

    # Just create an env by string name
    env = make_env("maze", rows=7, cols=7)
    appendage = make_appendage("dpad", hidden_dim=1152)
"""

from __future__ import annotations

from typing import Any

import torch

# ── Environment name map ──────────────────────────────────────────────────────

_ENV_MAP: dict[str, Any] = {}   # populated lazily to avoid circular imports

def _env_map() -> dict[str, Any]:
    global _ENV_MAP
    if _ENV_MAP:
        return _ENV_MAP
    from .environments.target_nav    import TargetNavEnvironment
    from .environments.spaceship     import SpaceshipNavEnvironment
    from .environments.grid_world    import GridWorldEnvironment
    from .environments.maze          import MazeEnvironment
    from .environments.button_task   import ButtonPressEnvironment, MCQButtonEnvironment
    from .environments.pointing      import PointingEnvironment
    from .environments.fruit_catcher import FruitCatcherEnvironment
    from .environments.treasure_hunt import TreasureHuntEnvironment
    from .environments.paint_canvas  import PaintCanvasEnvironment
    from .environments.whack_a_mole  import WhackAMoleEnvironment
    from .environments.mcq_navigator import MCQNavigatorEnvironment
    _ENV_MAP = {
        # canonical names
        "target_nav":    TargetNavEnvironment,
        "spaceship":     SpaceshipNavEnvironment,
        "grid_world":    GridWorldEnvironment,
        "maze":          MazeEnvironment,
        "button_press":  ButtonPressEnvironment,
        "mcq":           MCQButtonEnvironment,
        "pointing":      PointingEnvironment,
        "fruit_catcher": FruitCatcherEnvironment,
        "treasure_hunt": TreasureHuntEnvironment,
        "paint_canvas":  PaintCanvasEnvironment,
        "whack_a_mole":  WhackAMoleEnvironment,
        "mcq_navigator": MCQNavigatorEnvironment,
        # aliases
        "nav":           TargetNavEnvironment,
        "ship":          SpaceshipNavEnvironment,
        "grid":          GridWorldEnvironment,
        "button":        ButtonPressEnvironment,
        "paint":         PaintCanvasEnvironment,
        "mole":          WhackAMoleEnvironment,
        "fruit":         FruitCatcherEnvironment,
        "treasure":      TreasureHuntEnvironment,
    }
    return _ENV_MAP


# ── Appendage name map ────────────────────────────────────────────────────────

def _appendage_map() -> dict[str, Any]:
    from .appendages.joystick    import JoystickAppendage
    from .appendages.dpad        import DPadAppendage
    from .appendages.button      import ButtonAppendage, MultiButtonAppendage
    from .appendages.touchscreen import TouchscreenAppendage
    return {
        "joystick":     JoystickAppendage,
        "dpad":         DPadAppendage,
        "button":       ButtonAppendage,
        "multibutton":  MultiButtonAppendage,
        "multi_button": MultiButtonAppendage,
        "touchscreen":  TouchscreenAppendage,
        "touch":        TouchscreenAppendage,
    }


# ── Appendage → recommended environments ──────────────────────────────────────

#: Maps frozensets of appendage names to a list of (env_name, description) tuples.
APPENDAGE_ENV_MAP: dict[frozenset, list[tuple[str, str]]] = {
    # Single appendages
    frozenset({"joystick"}): [
        ("target_nav",  "Navigate a red dot to a blue target — pure joystick control"),
        ("spaceship",   "Fly a spaceship with Newtonian momentum physics"),
    ],
    frozenset({"dpad"}): [
        ("grid_world",  "Navigate a grid world with walls (guaranteed solvable)"),
        ("maze",        "Perfect-maze solver — requires backtracking"),
    ],
    frozenset({"button"}): [
        ("button_press","Press when the circle matches the target colour"),
    ],
    frozenset({"multibutton"}): [
        ("mcq",         "Multiple-choice visual questions (count dots, identify colour)"),
    ],
    frozenset({"touchscreen"}): [
        ("pointing",    "Tap a named coloured circle among distractors"),
        ("paint_canvas","Trace a shape by tapping waypoints in order"),
        ("whack_a_mole","Whack moles at random positions — avoid bombs"),
    ],
    # Combos
    frozenset({"joystick", "button"}): [
        ("fruit_catcher","Steer a basket left/right AND press to catch falling fruits"),
    ],
    frozenset({"dpad", "button"}): [
        ("treasure_hunt","Navigate a grid AND press to collect treasure tiles"),
    ],
    frozenset({"dpad", "multibutton"}): [
        ("mcq_navigator","Walk into the correct answer zone AND press the matching button"),
    ],
    frozenset({"joystick", "touchscreen"}): [
        ("pointing",    "Joystick-aimed cursor + touchscreen tap for fine confirmation"),
    ],
}


def recommended_envs(appendage_names: list[str] | set[str]) -> list[tuple[str, str]]:
    """
    Return a list of (env_name, description) pairs suited for the given appendages.

    Falls back to single-appendage recommendations when no combo match exists.

    Args:
        appendage_names: E.g. ["joystick", "button"] or {"dpad"}.

    Returns:
        List of (env_name, description) tuples.
    """
    key = frozenset(n.lower() for n in appendage_names)
    if key in APPENDAGE_ENV_MAP:
        return APPENDAGE_ENV_MAP[key]
    # Fallback: union of single-appendage recs
    recs = []
    for name in key:
        single = APPENDAGE_ENV_MAP.get(frozenset({name}), [])
        recs.extend(r for r in single if r not in recs)
    return recs or [("target_nav", "Default navigation environment")]


# ── Factories ─────────────────────────────────────────────────────────────────

def make_env(name: str, **kwargs) -> "BaseEnvironment":
    """
    Create an environment by string name.

    Args:
        name:    Environment name (see ``_ENV_MAP`` for all options).
        **kwargs: Passed to the environment constructor.

    Example::

        env = make_env("maze", rows=7, cols=7)
        env = make_env("fruit_catcher", n_fruits=3, max_steps=120)
    """
    em = _env_map()
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in em:
        raise KeyError(
            f"Unknown environment {name!r}. Available: {sorted(em.keys())}"
        )
    return em[key](**kwargs)


def make_appendage(
    name: str,
    hidden_dim: int,
    **kwargs,
) -> "BaseAppendage":
    """
    Create an action appendage by string name.

    Args:
        name:       Appendage name: "joystick", "dpad", "button",
                    "multibutton", "touchscreen".
        hidden_dim: VLM hidden state dimension.
        **kwargs:   Passed to the appendage constructor (e.g. n_buttons=4).

    Example::

        head = make_appendage("multibutton", hidden_dim=1152, n_buttons=4)
        head = make_appendage("touchscreen", hidden_dim=1152, vision_dim=1152)
    """
    am = _appendage_map()
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in am:
        raise KeyError(
            f"Unknown appendage {name!r}. Available: {sorted(am.keys())}"
        )
    return am[key](hidden_dim=hidden_dim, **kwargs)


def make_graft(
    vlm: torch.nn.Module,
    appendage_names: list[str] | str,
    hidden_dim: int | None = None,
    appendage_kwargs: dict[str, dict] | None = None,
    config=None,
):
    """
    Create a VLAGraft (single appendage) or CompositeGraft (multiple).

    Args:
        vlm:               HuggingFace VLM.
        appendage_names:   Single name string OR list of names.
        hidden_dim:        Auto-detected from VLM config if None.
        appendage_kwargs:  Per-appendage extra kwargs. E.g. {"multibutton": {"n_buttons": 4}}.
        config:            GraftConfig (auto-created if None).

    Returns:
        VLAGraft or CompositeGraft.

    Example::

        graft = make_graft(vlm, "joystick", hidden_dim=1152)
        graft = make_graft(vlm, ["joystick", "button"], hidden_dim=1152)
    """
    from .grafting.graft import GraftConfig, VLAGraft
    from .grafting.composite import CompositeGraft

    if isinstance(appendage_names, str):
        appendage_names = [appendage_names]

    if hidden_dim is None:
        cfg_obj = config or GraftConfig()
        hidden_dim = VLAGraft(vlm, make_appendage(appendage_names[0], hidden_dim=1),
                               cfg_obj)._detect_hidden_dim()

    ak = appendage_kwargs or {}
    appendages = {
        name: make_appendage(name, hidden_dim=hidden_dim, **ak.get(name, {}))
        for name in appendage_names
    }

    cfg = config or GraftConfig()
    if len(appendages) == 1:
        name, app = next(iter(appendages.items()))
        return VLAGraft(vlm=vlm, appendage=app, config=cfg)
    else:
        return CompositeGraft(vlm=vlm, appendages=appendages, config=cfg)


# ── Auto curriculum ───────────────────────────────────────────────────────────

def auto_curriculum(
    vlm: torch.nn.Module,
    processor,
    appendage_names: list[str] | str,
    hidden_dim: int | None = None,
    budget_steps: int = 1000,
    device: str | torch.device = "cpu",
    env_name: str | None = None,
    env_kwargs: dict | None = None,
    appendage_kwargs: dict | None = None,
    bc_fraction: float = 0.8,
    save_dir: str = "model_checkpoints/auto",
    verbose: bool = True,
) -> dict:
    """
    One-call training for any appendage combination.

    Automatically:
      1. Picks the best environment for the given appendages
      2. Builds the graft (VLAGraft or CompositeGraft)
      3. Allocates BC vs RL steps from ``budget_steps``
      4. Runs training and returns metrics

    Args:
        vlm:              HuggingFace VLM.
        processor:        Matching HuggingFace processor.
        appendage_names:  E.g. ["joystick", "button"] or "dpad".
        hidden_dim:       Auto-detected if None.
        budget_steps:     Total training steps budget.
        device:           Torch device string.
        env_name:         Override auto-selected environment.
        env_kwargs:       Extra kwargs for the environment constructor.
        appendage_kwargs: Per-appendage extra kwargs dict.
        bc_fraction:      Fraction of budget for BC (rest for RL).
        save_dir:         Checkpoint save directory.
        verbose:          Print progress.

    Returns:
        Dict with keys "bc" and/or "rl" containing metric lists,
        plus "graft" (the trained object) and "env".

    Example::

        results = auto_curriculum(
            vlm=vlm, processor=processor,
            appendage_names=["joystick", "button"],
            hidden_dim=1152,
            budget_steps=800,
            device="cuda",
        )
        graft = results["graft"]
    """
    if isinstance(appendage_names, str):
        appendage_names = [appendage_names]

    # Pick environment
    if env_name is None:
        recs = recommended_envs(appendage_names)
        env_name = recs[0][0]
        if verbose:
            print(f"[auto] Appendages: {appendage_names}")
            print(f"[auto] Selected env: {env_name}  ({recs[0][1]})")

    env = make_env(env_name, **(env_kwargs or {}))

    # Build graft
    graft = make_graft(
        vlm=vlm,
        appendage_names=appendage_names,
        hidden_dim=hidden_dim,
        appendage_kwargs=appendage_kwargs,
    )

    bc_steps = max(1, int(budget_steps * bc_fraction))
    rl_steps = budget_steps - bc_steps

    if verbose:
        print(f"[auto] Budget: {budget_steps} steps  (BC={bc_steps}, RL={rl_steps})")
        print(f"[auto] Graft: {type(graft).__name__}")

    results: dict = {"graft": graft, "env": env, "bc": [], "rl": []}

    from .grafting.graft import VLAGraft
    from .grafting.composite import CompositeGraft

    if isinstance(graft, VLAGraft):
        _run_single_graft(graft, processor, env, bc_steps, rl_steps,
                          device, save_dir, results)
    else:
        _run_composite_graft(graft, processor, env, bc_steps, rl_steps,
                             device, save_dir, results)

    return results


def _run_single_graft(graft, processor, env, bc_steps, rl_steps,
                      device, save_dir, results):
    """BC + RL training for a single-appendage VLAGraft."""
    from .training.trainer import BCTrainer, RLTrainer, TrainerConfig
    from .grafting.freezing import QUICK_CURRICULUM

    cfg = TrainerConfig(
        bc_steps=bc_steps,
        rl_steps=rl_steps,
        save_dir=save_dir,
        freezing_stages=QUICK_CURRICULUM,
    )

    bc = BCTrainer(graft, processor, env, config=cfg, device=device)
    results["bc"] = bc.train()

    if rl_steps > 0:
        rl = RLTrainer(graft, processor, env, config=cfg, device=device)
        results["rl"] = rl.train()


def _run_composite_graft(graft, processor, env, bc_steps, rl_steps,
                         device, save_dir, results):
    """
    BC training for a CompositeGraft.

    Runs one forward pass per sample and sums losses across all appendages.
    """
    from .training.trainer import _preprocess, _to_action_tensor
    from .grafting.freezing import QUICK_CURRICULUM, FreezingCurriculum
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    import random as _random
    import time

    dev = torch.device(device)
    graft.to(dev)
    graft.train()

    curriculum = FreezingCurriculum(
        vlm=graft.vlm,
        stages=QUICK_CURRICULUM,
    )
    optimizer = optim.AdamW(
        graft.parameter_groups(appendage_lr=1e-4, vlm_lr=1e-6),
        weight_decay=1e-4,
    )

    metrics = []
    print(f"[auto] Composite BC training: {bc_steps} steps")
    start = time.time()

    for step in tqdm(range(bc_steps), desc="Composite BC"):
        curriculum.step(step)
        graft.train()

        obs = env.reset()
        for _ in range(_random.randint(0, 6)):
            expert_act = env.expert_action()
            r = env.step(expert_act)
            obs = r.observation
            if r.done:
                obs = env.reset()
                break

        expert_act = env.expert_action()
        inputs = _preprocess(processor, obs, env.prompt, dev)

        optimizer.zero_grad()
        out = graft(**inputs)

        total_loss = torch.tensor(0.0, device=dev)
        for name, app in graft.appendages.items():
            pred = out[name]   # [1, *action_shape]
            # Expert action for this appendage
            if isinstance(expert_act, dict):
                ea = expert_act.get(name, expert_act.get(name.split("_")[0], [0.0]))
            else:
                ea = expert_act
            target = _to_action_tensor(app, [ea], dev)
            total_loss = total_loss + app.action_loss(pred, target)

        total_loss.backward()
        nn.utils.clip_grad_norm_(list(graft.appendages.parameters()), 1.0)
        optimizer.step()

        m = {"step": step, "bc/loss": float(total_loss)}
        metrics.append(m)
        if step % 100 == 0:
            print(f"  step {step:5d}  loss={float(total_loss):.4f}  t={time.time()-start:.0f}s")

    results["bc"] = metrics
    graft.save(save_dir)
