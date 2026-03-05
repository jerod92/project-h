# vla-hands

**Give VLMs hands.** Graft lightweight action heads onto any Vision-Language Model
to create a Vision-Language-Action (VLA) model — without touching the VLM backbone.

```
  image + prompt
       │
  ┌────▼─────────────────────────────────┐
  │           VLM Backbone               │
  │  vision encoder ──► [patch tokens]   │
  │  text embeds    ──► [text tokens]    │
  │                       transformer    │
  │                        layers…       │
  │                          │           │
  │                   last hidden state  │
  └────────────────────┬─────────────────┘
                       │   (one forward pass)
            ┌──────────┼──────────┐
            ▼          ▼          ▼
        [LM head]  [Joystick]  [Button]   ← tiny MLPs
         tokens    (x, y)∈[-1,1]²  {0,1}
```

The action heads ("appendages") fire in parallel with token generation.
The VLM backbone is structurally unchanged — only the appendage MLP weights are saved.

---

## Appendages

| Appendage | Output | Loss | Best for |
|-----------|--------|------|----------|
| `JoystickAppendage` | `(x, y) ∈ [-1, 1]²` | Huber | Navigation, continuous steering |
| `DPadAppendage` | `{STAY, UP, DOWN, LEFT, RIGHT}` | Cross-entropy | Grid navigation, menus |
| `ButtonAppendage` | `press ∈ [0, 1]` | BCE | Binary decisions, timed actions |
| `MultiButtonAppendage` | `[p₀…pₙ] ∈ [0,1]ⁿ` | BCE per button | MCQ, multi-choice interaction |
| `TouchscreenAppendage` | `(x, y) ∈ [0, 1]²` | Huber | Pointing, tapping, clicking |

The `TouchscreenAppendage` optionally receives a **vision skip connection** —
mean-pooled patch embeddings from the VLM's vision encoder — for more spatially
precise coordinate prediction.

Add a new appendage: subclass `BaseAppendage`, implement `forward()`, `action_loss()`,
and `action_spec`. That's it.

---

## Environments

### Single-appendage

| Environment | Appendage | Task |
|-------------|-----------|------|
| `TargetNavEnvironment` | Joystick | Move red agent to blue target |
| `SpaceshipNavEnvironment` | Joystick | Newtonian physics — account for momentum |
| `GridWorldEnvironment` | DPad | Navigate grid with walls |
| `MazeEnvironment` | DPad | Perfect maze — backtracking required |
| `ButtonPressEnvironment` | Button | Press iff circle matches target colour |
| `MCQButtonEnvironment` | MultiButton | Visual multiple-choice (count dots, ID colour) |
| `PointingEnvironment` | Touchscreen | Tap named coloured circle among distractors |
| `PaintCanvasEnvironment` | Touchscreen | Tap waypoints in order to trace a shape |
| `WhackAMoleEnvironment` | Touchscreen | Hit moles, avoid bombs; prioritise gold ones |

### Multi-appendage (CompositeGraft)

| Environment | Appendages | Task |
|-------------|-----------|------|
| `FruitCatcherEnvironment` | Joystick + Button | Steer basket L/R **and** press to catch falling fruits |
| `TreasureHuntEnvironment` | DPad + Button | Navigate grid **and** press to collect treasure tiles |
| `MCQNavigatorEnvironment` | DPad + MultiButton | Walk to answer zone **and** press matching button |

All environments include:
- Deterministic **expert policy** (for BC warm-start)
- **Dynamic prompt vocabulary** — 6 synonym phrasings re-sampled every `reset()`
- PIL-only rendering (no pygame, no display required)

---

## Quick Start

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from vla_hands import VLAGraft, GraftConfig, JoystickAppendage
from vla_hands import TargetNavEnvironment, TrainingCurriculum, CurriculumConfig

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
vlm = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

hidden_dim = vlm.config.text_config.hidden_size   # 1152 for SmolVLM-256M

# Graft a joystick onto the VLM
graft = VLAGraft(vlm=vlm, appendage=JoystickAppendage(hidden_dim))

# Train: BC warm-start → RL fine-tuning
env = TargetNavEnvironment()
curriculum = TrainingCurriculum(
    graft, processor, env,
    CurriculumConfig(bc_steps=500, rl_steps=200, device="cuda"),
)
metrics = curriculum.run()

# Save appendage weights only (~100 KB)
graft.save("model_checkpoints/joystick_nav")

# One-line inference reload
from vla_hands.training.trainer import _preprocess
obs = env.reset()
inputs = _preprocess(processor, obs, env.prompt, "cuda")
action = graft.predict_action(**inputs)
print(graft.appendage.decode(action))   # JoystickAction(x=+0.71, y=-0.24)
```

### Automated training (any appendage combo)

```python
from vla_hands import auto_curriculum, recommended_envs

# See what environments work for your appendages
recommended_envs(["joystick", "button"])
# → [("fruit_catcher", "Steer a basket left/right AND press to catch falling fruits")]

# One call: selects env, builds graft, runs BC+RL, returns metrics
results = auto_curriculum(
    vlm=vlm, processor=processor,
    appendage_names=["joystick", "button"],
    hidden_dim=hidden_dim,
    budget_steps=800,
    device="cuda",
)
graft = results["graft"]
```

### Multi-appendage (CompositeGraft)

```python
from vla_hands import CompositeGraft, JoystickAppendage, ButtonAppendage
from vla_hands.environments.fruit_catcher import FruitCatcherEnvironment

graft = CompositeGraft(
    vlm=vlm,
    appendages={
        "joystick": JoystickAppendage(hidden_dim),
        "button":   ButtonAppendage(hidden_dim),
    },
)

# One forward pass → two action tensors
env = FruitCatcherEnvironment()
obs = env.reset()
inputs = _preprocess(processor, obs, env.prompt, "cuda")
out = graft(**inputs)
# → {"joystick": Tensor[1,2], "button": Tensor[1,1], "lm_logits": ..., ...}

graft.save("model_checkpoints/fruit_catcher/")
```

### Touchscreen with vision skip connection

```python
from vla_hands import TouchscreenAppendage, VLAGraft

vision_dim = VLAGraft.detect_vision_dim(vlm)   # reads from model config
head = TouchscreenAppendage(hidden_dim=hidden_dim, vision_dim=vision_dim)

# VLAGraft auto-registers a forward hook on the vision encoder
graft = VLAGraft(vlm=vlm, appendage=head)
# now each forward pass: VLM hidden state + vision encoder patches → (x, y)
```

### LoRA (parameter-efficient fine-tuning)

```python
from vla_hands import apply_lora, LoRAConfig

vlm = apply_lora(vlm, LoRAConfig(r=8, alpha=16))  # wraps vlm in peft.PeftModel
# VLAGraft / CompositeGraft usage is identical — interface unchanged
# Trains ~0.5% of VLM params instead of 100%
```

---

## Installation

```bash
git clone https://github.com/jerod92/vla-hands
cd vla-hands
pip install -e .

# Optional training extras (wandb, tensorboard, accelerate)
pip install -e ".[train]"

# LoRA support
pip install peft>=0.10.0
```

**Core dependencies:** `torch >= 2.1`, `transformers >= 4.40`, `Pillow`, `numpy`, `tqdm`

---

## Training Design

### Output timing
The action head fires on **every forward pass** that includes visual input, in
parallel with token generation. It reads the **last non-padding token's hidden
state** — which has attended over all image and text tokens.

In rollouts: discard token logits, use the action.
In language tasks: discard the action, use the tokens.
Cost: a tiny MLP on top of an already-run VLM.

### Staged freezing curriculum

```
Stage 0  (step    0)  appendage only    — VLM fully frozen
Stage 1  (step  500)  last 2 layers     — unfreezes from output end backwards
Stage 2  (step 1500)  last 6 layers
Stage 3* (step 4000)  full fine-tune    — optional, risks forgetting
```

Unfrozen VLM layers use `vlm_lr ≈ 1e-6` (100× smaller than appendage LR).
The language generation path is untouched throughout.

Use `QUICK_CURRICULUM` (appendage-only) for fast demos.
Use `DEFAULT_CURRICULUM` for production training.

### Training phases

1. **Behavioral Cloning (BC)** — supervised from the built-in expert policy. Fast
   convergence; teaches the head to read VLM features.
2. **REINFORCE RL** — fine-tunes with actual environment rewards after BC warm-start.

---

## Utilities

### GIF recorder

```python
from vla_hands import save_rollout_gif, record_expert_gif

save_rollout_gif(graft, processor, env, "demo.gif", n_steps=30, fps=6)
record_expert_gif(env, "expert.gif", n_steps=20)
```

### Training visualisation

```python
from vla_hands import plot_training_curves, TrainingSummary

metrics = curriculum.run()
plot_training_curves(metrics)                      # BC loss + eval success rate
print(TrainingSummary.from_metrics(metrics))       # formatted summary table
```

### String factories

```python
from vla_hands import make_env, make_appendage, make_graft

env  = make_env("maze", rows=7, cols=7)
head = make_appendage("multibutton", hidden_dim=1152, n_buttons=4)
graft = make_graft(vlm, ["dpad", "button"], hidden_dim=1152)
```

### One-line load for inference

```python
graft = VLAGraft.from_pretrained(
    vlm_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    appendage=JoystickAppendage(hidden_dim=1152),
    checkpoint_path="model_checkpoints/bc_final",
    device="cuda",
)
```

---

## Model Compatibility

Works with any HuggingFace model that:
- Accepts `pixel_values + input_ids`
- Supports `output_hidden_states=True`

**Verified:** SmolVLM (256M, 500M, 2B Instruct variants) — full Colab demo available.

**Expected to work** (API-compatible, not yet validated):
LLaVA-NeXT, Qwen-VL, InstructBLIP, PaliGemma.

Checkpoint size: only the appendage MLP weights are saved (~50 KB – 2 MB).
The VLM backbone is loaded separately from HuggingFace Hub.

---

## Training Scripts

```bash
python scripts/train_joystick.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --bc-steps 1000 --rl-steps 300 --device cuda

python scripts/train_dpad.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --bc-steps 1000 --grid-size 8 --device cuda
```

---

## Benchmarking

```python
from vla_hands import BenchmarkSuite, run_expert_baseline

# Expert upper bound
run_expert_baseline(env, n_episodes=50)

# Evaluate a trained graft
suite = BenchmarkSuite(graft, processor, [env], device="cuda")
result = suite.run_benchmark(env, n_episodes=50)
print(f"success={result.success_rate:.0%}  reward={result.mean_reward:+.1f}")

# Compare multiple grafts side-by-side
from vla_hands import compare_grafts
compare_grafts({"joystick_bc": g1, "joystick_rl": g2}, processor, env)
```

---

## Project Layout

```
vla_hands/
├── appendages/          Action heads
│   ├── base.py          BaseAppendage + ActionSpec
│   ├── joystick.py      Continuous (x, y) ∈ [-1,1]²
│   ├── dpad.py          Discrete 5-way
│   ├── button.py        Binary + N-button
│   └── touchscreen.py   Absolute (x, y) ∈ [0,1]² with vision skip
│
├── environments/        Training tasks
│   ├── base.py          BaseEnvironment + EnvStepResult
│   ├── prompt_vocab.py  Dynamic synonym sets (PromptVocab)
│   ├── target_nav.py    Joystick: move to target
│   ├── spaceship.py     Joystick: Newtonian physics
│   ├── grid_world.py    DPad: grid with walls
│   ├── maze.py          DPad: perfect maze
│   ├── button_task.py   Button / MultiButton: colour match + MCQ
│   ├── pointing.py      Touchscreen: tap named circle
│   ├── paint_canvas.py  Touchscreen: trace path waypoints
│   ├── whack_a_mole.py  Touchscreen: hit moles, avoid bombs
│   ├── fruit_catcher.py Joystick + Button: navigate + catch
│   ├── treasure_hunt.py DPad + Button: navigate + collect
│   └── mcq_navigator.py DPad + MultiButton: navigate + select
│
├── grafting/            VLM ↔ appendage bridge
│   ├── graft.py         VLAGraft (single appendage)
│   ├── composite.py     CompositeGraft (N appendages, one VLM pass)
│   ├── freezing.py      FreezingCurriculum (staged unfreezing)
│   └── lora.py          LoRA via peft (apply_lora, merge_lora)
│
├── training/
│   ├── trainer.py       BCTrainer + RLTrainer
│   └── curriculum.py    TrainingCurriculum (BC → RL pipeline)
│
├── utils/
│   ├── gif.py           save_rollout_gif, record_expert_gif
│   └── viz.py           plot_training_curves, TrainingSummary
│
├── benchmarks/
│   └── suite.py         BenchmarkSuite, run_expert_baseline, compare_grafts
│
├── auto.py              make_env / make_appendage / auto_curriculum
│
notebooks/
└── quickstart_colab.ipynb   Full demo on Colab/Kaggle (SmolVLM-256M)
```

---

## Roadmap

- [ ] WandB / TensorBoard logging in trainers
- [ ] Gymnasium-compatible environment wrapper
- [ ] ONNX / TorchScript export for deployment
- [ ] Graft zoo (downloadable checkpoints on HF Hub)
- [ ] Multi-task training (one graft, multiple environments simultaneously)
- [ ] 6-DOF continuous control appendage

---

## License

MIT
