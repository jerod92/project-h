# vla-hands

**Give VLMs action capabilities by grafting lightweight "appendage" heads onto them.**

`vla-hands` turns a frozen Vision-Language Model (VLM) into a Vision-Language-Action
model (VLA) by attaching a small MLP "appendage" to the VLM's hidden states.
The appendage fires in parallel with token generation on every forward pass,
producing structured actions (joystick, d-pad, button, pointer…) alongside text.

```
        ┌─────────────────────────────────┐
        │          VLM Backbone           │
        │  (frozen by default)            │
        │                                 │
  image ──► vision encoder ──► [tokens]  │
  text  ──► token embeds   ──►  ┊        │
                                 ▼        │
                           transformer   │
                            layers…      │
                                 │        │
                        last hidden state │
                          /          \   │
                         /            \  │
              ┌──────────┐         ┌───────┐
              │ LM head  │         │Action │
              │ (tokens) │         │ Head  │
              └──────────┘         └───────┘
               "Hello"            (x=0.7, y=-0.2)
```

---

## Concepts

| Term | Meaning |
|------|---------|
| **Appendage** | A small MLP grafted onto the VLM's hidden states to produce actions |
| **Graft** | The combined (VLM + appendage) module — `VLAGraft` |
| **Environment** | A task (renders images, gives rewards, has an expert policy) |
| **Curriculum** | Staged training: BC warm-start → RL fine-tuning + gradual unfreezing |

---

## Included Appendages

| Appendage | Action space | Use cases |
|-----------|-------------|-----------|
| `JoystickAppendage` | `(x, y) ∈ [-1, 1]²` continuous | Navigation, pointing, camera control |
| `DPadAppendage` | `{STAY, UP, DOWN, LEFT, RIGHT}` discrete | Grid nav, menus, turn-based games |

Adding a new appendage: subclass `BaseAppendage`, implement `forward()`, `action_loss()`, and `action_spec`.

---

## Included Environments

| Environment | Paired with | Task |
|------------|-------------|------|
| `TargetNavEnvironment` | Joystick | Move red agent to blue target |
| `GridWorldEnvironment` | D-pad | Navigate grid, avoid walls, reach goal |

---

## Design Decisions

### 1. Output timing
The action head fires on **every forward pass** that includes visual input —
in parallel with token generation. It uses the **last non-padding token's hidden state**
as its input feature (this position has attended over all image + text tokens).

In episode rollouts the caller discards token logits; in language tasks the caller
discards the action. The head is always "live" — it adds only a tiny MLP overhead.

### 2. Freezing strategy
Training proceeds in discrete stages (not a continuous thaw):

```
Stage 0  (step 0)    : All VLM weights frozen — train action head only.
Stage 1  (step 500)  : Unfreeze final 2 transformer layers.
Stage 2  (step 1500) : Unfreeze final 6 transformer layers.
Stage 3* (step 4000) : Full fine-tuning (optional — risks forgetting).
```

Layers unfreeze from the **output end backwards** (closest to the action head first).
The early layers (embeddings, early transformer blocks) stay frozen unless you
explicitly opt into `STAGE_FULL`. The language generation path is never removed
or altered — we add a branch, not a replacement.

Each unfrozen VLM layer uses `vlm_lr ≈ 1e-6` (100× smaller than the appendage LR)
to prevent catastrophic forgetting.

### 3. Model compatibility
Works with any HuggingFace model that:
- Accepts `pixel_values` + `input_ids`
- Supports `output_hidden_states=True`

Tested families: **LLaVA**, **SmolVLM**, **Qwen-VL**, **InstructBLIP**.
The VLM backbone is structurally unchanged. Only appendage weights are saved in
the graft checkpoint (MBs, not GBs).

---

## Quick Start

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from vla_hands import (
    VLAGraft, GraftConfig,
    JoystickAppendage,
    TargetNavEnvironment,
    TrainingCurriculum, CurriculumConfig,
)

# Load any HuggingFace VLM
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
vlm = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

# Graft a joystick onto the VLM
appendage = JoystickAppendage(hidden_dim=vlm.config.hidden_size)
graft = VLAGraft(vlm=vlm, appendage=appendage)

# Train on the target-navigation environment
env = TargetNavEnvironment()
curriculum = TrainingCurriculum(
    graft, processor, env,
    CurriculumConfig(bc_steps=500, rl_steps=200, device="cuda"),
)
curriculum.run()

# Save the appendage (small checkpoint)
graft.save("checkpoints/my_joystick")

# Inference
import torch
obs = env.reset()
inputs = processor(images=obs, text=env.prompt, return_tensors="pt")
action = graft.predict_action(**inputs)
joystick = appendage.decode(action)
print(joystick)  # JoystickAction(x=+0.712, y=-0.241)
```

---

## Installation

```bash
git clone https://github.com/your-org/vla-hands
cd vla-hands
pip install -e ".[train]"
```

**Dependencies:** `torch >= 2.1`, `transformers >= 4.40`, `Pillow`, `numpy`, `tqdm`

Optional: `wandb`, `tensorboard`, `accelerate` (install with `pip install -e ".[train]"`)

---

## Training Scripts

```bash
# Joystick — target navigation
python scripts/train_joystick.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --bc-steps 1000 --rl-steps 300 --device cuda

# D-pad — grid world navigation
python scripts/train_dpad.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --bc-steps 1000 --grid-size 8 --device cuda
```

---

## Benchmarking

```python
from vla_hands import BenchmarkSuite, run_expert_baseline, TargetNavEnvironment

env = TargetNavEnvironment()

# Expert upper bound
run_expert_baseline(env, n_episodes=50)

# Evaluate trained graft
suite = BenchmarkSuite(graft, processor, [env], device="cuda")
results = suite.run_all(n_episodes=50)
```

---

## Extending: Adding a New Appendage

```python
from vla_hands.appendages.base import BaseAppendage, ActionSpec
import torch, torch.nn as nn

class ButtonAppendage(BaseAppendage):
    """Single binary button (press / no-press)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._spec = ActionSpec(name="button", shape=(1,), dtype="continuous",
                                low=0.0, high=1.0)

    @property
    def action_spec(self): return self._spec

    def forward(self, hidden_state):
        return self.net(hidden_state)

    def action_loss(self, predicted, target, weights=None):
        return torch.nn.functional.binary_cross_entropy(predicted, target)
```

---

## Roadmap

- [ ] Pointer appendage (2D screen coordinate)
- [ ] Multi-axis continuous control (6-DOF robot arm)
- [ ] Composite appendages (multiple action heads per graft)
- [ ] WandB / TensorBoard logging integration
- [ ] ONNX export for deployment
- [ ] Gymnasium-compatible environment wrapper
- [ ] Pre-trained appendage zoo (downloadable checkpoints)

---

## License

MIT
