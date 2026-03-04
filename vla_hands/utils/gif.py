"""
Animated GIF recorder for VLA episode rollouts.

Records a full episode (or any sequence of PIL Images) and saves an animated
GIF.  Useful for debugging trained grafts, sharing results, and generating
demos without needing a display or video encoder.

Usage::

    from vla_hands.utils import save_rollout_gif
    from vla_hands.training.trainer import _preprocess

    # Using a trained graft
    save_rollout_gif(
        graft, processor, env, path="demo.gif",
        n_steps=30, seed=0, device="cuda", fps=6,
    )

    # From a raw list of PIL images
    save_rollout_gif(frames=my_frames, path="raw.gif", fps=12)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_rollout_gif(
    graft=None,
    processor=None,
    env=None,
    path: str | Path = "rollout.gif",
    frames: list | None = None,
    n_steps: int = 40,
    seed: int | None = 0,
    device: str | torch.device = "cpu",
    fps: int = 6,
    scale: int = 1,
    loop: int = 0,
) -> Path:
    """
    Record a graft episode and save as an animated GIF.

    Either pass ``(graft, processor, env)`` for automatic rollout, or pass a
    pre-collected ``frames`` list of PIL Images.

    Args:
        graft:     Trained VLAGraft (or CompositeGraft).
        processor: HuggingFace processor matching the VLM.
        env:       Environment instance with reset()/step()/expert_action().
        path:      Output file path (must end in .gif).
        frames:    Pre-collected list of PIL Images (skips rollout if given).
        n_steps:   Maximum number of steps per episode.
        seed:      Random seed passed to env.reset().
        device:    Torch device.
        fps:       Frames per second in the output GIF.
        scale:     Integer upscale factor (1 = no scaling).
        loop:      Number of loops (0 = infinite).

    Returns:
        Resolved Path of the saved GIF.
    """
    from PIL import Image

    if frames is None:
        if graft is None or processor is None or env is None:
            raise ValueError(
                "Pass either frames= (list of PIL Images) or "
                "(graft, processor, env) for automatic rollout."
            )
        frames = _run_rollout(graft, processor, env, n_steps, seed, device)

    if not frames:
        raise ValueError("No frames to save.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally upscale for visibility
    if scale > 1:
        frames = [
            f.resize((f.width * scale, f.height * scale), Image.NEAREST)
            for f in frames
        ]

    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )
    print(f"[GIF] Saved {len(frames)} frames → {path}  ({path.stat().st_size // 1024} KB)")
    return path


def _run_rollout(graft, processor, env, n_steps: int, seed, device) -> list:
    """Collect frames from one episode using the trained graft."""
    from ..training.trainer import _preprocess, _select_action

    graft_device = torch.device(device)
    graft.eval()

    obs = env.reset(seed=seed)
    frames = [obs]

    for _ in range(n_steps):
        with torch.no_grad():
            inputs = _preprocess(processor, obs, env.prompt, graft_device)

            # Support both VLAGraft and CompositeGraft
            out = graft(**inputs)

        # Extract action — out may be a dict of tensors (CompositeGraft)
        #                  or a plain dict with "action" key (VLAGraft)
        if "action" in out:
            raw = out["action"]
            action_val = _select_action(graft.appendage, raw)
        else:
            # CompositeGraft: build full action dict for multi-appendage envs
            action_val = {
                name: _select_action(app, out[name])
                for name, app in graft.appendages.items()  # type: ignore[union-attr]
                if name in out
            }

        result = env.step(action_val)
        frames.append(result.observation)
        obs = result.observation
        if result.done:
            break

    return frames


def record_expert_gif(
    env,
    path: str | Path = "expert.gif",
    n_steps: int = 40,
    seed: int = 0,
    fps: int = 6,
    scale: int = 1,
) -> Path:
    """
    Record the expert policy (no graft needed) and save as GIF.

    Useful for verifying environments and generating reference demos.
    """
    obs = env.reset(seed=seed)
    frames = [obs]
    for _ in range(n_steps):
        action = env.expert_action()
        result = env.step(action)
        frames.append(result.observation)
        if result.done:
            break
    return save_rollout_gif(frames=frames, path=path, fps=fps, scale=scale)
