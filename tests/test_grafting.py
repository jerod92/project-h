"""Tests for VLAGraft, CompositeGraft, FreezingCurriculum, and LoRA."""

import pytest
import torch
import torch.nn as nn

from vla_hands.appendages.joystick import JoystickAppendage
from vla_hands.appendages.dpad import DPadAppendage
from vla_hands.appendages.button import ButtonAppendage, MultiButtonAppendage
from vla_hands.appendages.touchscreen import TouchscreenAppendage
from vla_hands.grafting.graft import GraftConfig, VLAGraft
from vla_hands.grafting.composite import CompositeGraft
from vla_hands.grafting.freezing import (
    FreezingCurriculum,
    FreezingStage,
    STAGE_APPENDAGE_ONLY,
    STAGE_LAST_2,
)


# ── Minimal stub VLM (no HuggingFace download) ───────────────────────────────

HIDDEN_DIM = 64
VISION_DIM = 32


class _FakeVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(VISION_DIM, VISION_DIM)

    def forward(self, pixel_values=None, **kw):
        batch = pixel_values.shape[0] if pixel_values is not None else 1
        last_hidden = torch.randn(batch, 16, VISION_DIM)
        # Return object with last_hidden_state attribute (like HF vision models)
        class _Out:
            pass
        o = _Out()
        o.last_hidden_state = last_hidden
        return o


class _FakeHiddenStates:
    def __init__(self, hidden_size: int, seq_len: int, batch: int):
        h = torch.randn(batch, seq_len, hidden_size)
        self.hidden_states = (h, h, h)
        self.logits = torch.randn(batch, seq_len, 100)
        self.loss = None


class _FakeVLMConfig:
    hidden_size = HIDDEN_DIM
    text_config = None


class _FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeVLMConfig()
        self.layers = nn.ModuleList([nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(4)])
        self.model = type("M", (), {"layers": self.layers})()
        # Expose a vision_model for hook tests
        self.vision_model = _FakeVisionEncoder()

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None,
                labels=None, output_hidden_states=False, return_dict=False, **kw):
        batch = input_ids.shape[0] if input_ids is not None else 2
        seq = 5
        # Trigger the vision model forward if pixel_values provided
        if pixel_values is not None:
            self.vision_model(pixel_values=pixel_values)
        return _FakeHiddenStates(HIDDEN_DIM, seq, batch)


# ── VLAGraft ─────────────────────────────────────────────────────────────────

class TestVLAGraft:
    def setup_method(self):
        self.vlm = _FakeVLM()
        self.app = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        self.graft = VLAGraft(
            vlm=self.vlm,
            appendage=self.app,
            config=GraftConfig(feature_extraction="last", hidden_dim=HIDDEN_DIM),
        )

    def test_forward_returns_keys(self):
        out = self.graft(
            pixel_values=torch.zeros(2, 3, 32, 32),
            input_ids=torch.zeros(2, 5, dtype=torch.long),
            attention_mask=torch.ones(2, 5),
        )
        assert "lm_logits" in out
        assert "action" in out
        assert "action_features" in out

    def test_action_shape(self):
        out = self.graft(
            input_ids=torch.zeros(2, 5, dtype=torch.long),
            attention_mask=torch.ones(2, 5),
        )
        assert out["action"].shape == (2, 2)

    def test_predict_action_no_grad(self):
        action = self.graft.predict_action(
            input_ids=torch.zeros(1, 5, dtype=torch.long),
            attention_mask=torch.ones(1, 5),
        )
        assert action.shape == (1, 2)
        assert not action.requires_grad

    def test_feature_extraction_last(self):
        graft = VLAGraft(self.vlm, self.app, GraftConfig("last", HIDDEN_DIM))
        out = graft(input_ids=torch.zeros(2, 5, dtype=torch.long),
                    attention_mask=torch.ones(2, 5))
        assert out["action_features"].shape == (2, HIDDEN_DIM)

    def test_feature_extraction_first(self):
        graft = VLAGraft(self.vlm, self.app, GraftConfig("first", HIDDEN_DIM))
        out = graft(input_ids=torch.zeros(2, 5, dtype=torch.long))
        assert out["action_features"].shape == (2, HIDDEN_DIM)

    def test_feature_extraction_mean(self):
        graft = VLAGraft(self.vlm, self.app, GraftConfig("mean", HIDDEN_DIM))
        out = graft(input_ids=torch.zeros(2, 5, dtype=torch.long),
                    attention_mask=torch.ones(2, 5))
        assert out["action_features"].shape == (2, HIDDEN_DIM)

    def test_save_and_load(self, tmp_path):
        self.graft.save(tmp_path / "test_graft")
        assert (tmp_path / "test_graft" / "graft_config.json").exists()
        assert (tmp_path / "test_graft" / "appendage.pt").exists()

        graft2 = VLAGraft(
            vlm=_FakeVLM(),
            appendage=JoystickAppendage(HIDDEN_DIM),
            config=GraftConfig(hidden_dim=HIDDEN_DIM),
        )
        graft2.load_appendage(tmp_path / "test_graft")

    def test_parameter_groups(self):
        groups = self.graft.parameter_groups(appendage_lr=1e-4, vlm_lr=1e-6)
        assert len(groups) >= 1
        assert groups[0]["lr"] == 1e-4

    def test_repr(self):
        s = repr(self.graft)
        assert "VLAGraft" in s
        assert "JoystickAppendage" in s


class TestDPadGraft:
    def test_dpad_action_shape(self):
        vlm = _FakeVLM()
        app = DPadAppendage(hidden_dim=HIDDEN_DIM)
        graft = VLAGraft(vlm, app, GraftConfig(hidden_dim=HIDDEN_DIM))
        out = graft(input_ids=torch.zeros(2, 5, dtype=torch.long))
        assert out["action"].shape == (2, 5)


class TestVisionHookGraft:
    """VLAGraft with TouchscreenAppendage should register a vision hook."""

    def test_hook_registered(self):
        vlm = _FakeVLM()
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM, vision_dim=VISION_DIM)
        graft = VLAGraft(vlm, app, GraftConfig(hidden_dim=HIDDEN_DIM))
        assert graft._vision_hook is not None

    def test_hook_also_registered_without_vision_proj(self):
        """Hook is registered whenever needs_vision_features=True, even if vision_proj is None."""
        vlm = _FakeVLM()
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM, vision_dim=None)
        graft = VLAGraft(vlm, app, GraftConfig(hidden_dim=HIDDEN_DIM))
        # vision_proj is None but hook is still registered (features simply not used)
        assert app.vision_proj is None
        assert graft._vision_hook is not None  # hook registered; no-op if no vision_proj

    def test_forward_succeeds_with_hook(self):
        vlm = _FakeVLM()
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM, vision_dim=VISION_DIM)
        graft = VLAGraft(vlm, app, GraftConfig(hidden_dim=HIDDEN_DIM))
        out = graft(
            pixel_values=torch.zeros(2, 3, 32, 32),
            input_ids=torch.zeros(2, 5, dtype=torch.long),
        )
        assert "action" in out
        assert out["action"].shape == (2, 2)


# ── CompositeGraft ────────────────────────────────────────────────────────────

class TestCompositeGraft:
    def setup_method(self):
        self.vlm = _FakeVLM()
        self.appendages = {
            "joystick": JoystickAppendage(HIDDEN_DIM),
            "button": ButtonAppendage(HIDDEN_DIM),
        }
        self.graft = CompositeGraft(
            vlm=self.vlm,
            appendages=self.appendages,
            config=GraftConfig(feature_extraction="last", hidden_dim=HIDDEN_DIM),
        )

    def test_forward_returns_all_keys(self):
        out = self.graft(
            input_ids=torch.zeros(2, 5, dtype=torch.long),
            attention_mask=torch.ones(2, 5),
        )
        assert "joystick" in out
        assert "button" in out
        assert "lm_logits" in out
        assert "action_features" in out

    def test_joystick_shape(self):
        out = self.graft(input_ids=torch.zeros(2, 5, dtype=torch.long))
        assert out["joystick"].shape == (2, 2)

    def test_button_shape(self):
        out = self.graft(input_ids=torch.zeros(2, 5, dtype=torch.long))
        assert out["button"].shape == (2, 1)

    def test_predict_actions_no_grad(self):
        actions = self.graft.predict_actions(
            input_ids=torch.zeros(1, 5, dtype=torch.long)
        )
        assert "joystick" in actions
        assert not actions["joystick"].requires_grad

    def test_save_and_load(self, tmp_path):
        self.graft.save(tmp_path / "composite")
        assert (tmp_path / "composite" / "composite_config.json").exists()
        assert (tmp_path / "composite" / "appendage_joystick.pt").exists()
        assert (tmp_path / "composite" / "appendage_button.pt").exists()

    def test_three_appendages(self):
        vlm = _FakeVLM()
        graft = CompositeGraft(
            vlm=vlm,
            appendages={
                "joy": JoystickAppendage(HIDDEN_DIM),
                "btn": ButtonAppendage(HIDDEN_DIM),
                "multi": MultiButtonAppendage(HIDDEN_DIM, n_buttons=4),
            },
            config=GraftConfig(hidden_dim=HIDDEN_DIM),
        )
        out = graft(input_ids=torch.zeros(2, 5, dtype=torch.long))
        assert out["joy"].shape == (2, 2)
        assert out["btn"].shape == (2, 1)
        assert out["multi"].shape == (2, 4)

    def test_joint_backward(self):
        """Loss sum across appendages should be differentiable."""
        out = self.graft(input_ids=torch.zeros(2, 5, dtype=torch.long))
        loss = (
            self.appendages["joystick"].action_loss(out["joystick"], torch.zeros(2, 2))
            + self.appendages["button"].action_loss(out["button"], torch.zeros(2, 1))
        )
        loss.backward()  # must not raise


# ── FreezingCurriculum ────────────────────────────────────────────────────────

class TestFreezingCurriculum:
    def setup_method(self):
        self.vlm = _FakeVLM()

    def test_initial_all_frozen(self):
        curriculum = FreezingCurriculum(self.vlm, stages=[STAGE_APPENDAGE_ONLY])
        for p in self.vlm.parameters():
            assert not p.requires_grad

    def test_stage_0_no_change(self):
        curriculum = FreezingCurriculum(
            self.vlm, stages=[STAGE_APPENDAGE_ONLY, STAGE_LAST_2]
        )
        changed = curriculum.step(0)
        assert changed

    def test_stage_change(self):
        stage1 = FreezingStage("s1", "desc", n_layers_unfrozen=0, start_step=0)
        stage2 = FreezingStage("s2", "desc", n_layers_unfrozen=2, start_step=10)
        curriculum = FreezingCurriculum(self.vlm, stages=[stage1, stage2])
        curriculum.step(0)
        curriculum.step(9)
        changed = curriculum.step(10)
        assert changed

    def test_frozen_count(self):
        curriculum = FreezingCurriculum(self.vlm, stages=[STAGE_APPENDAGE_ONLY])
        curriculum.step(0)
        n_trainable, n_total = curriculum.frozen_param_count()
        assert n_trainable == 0
        assert n_total > 0


# ── LoRA (graceful skip if peft not installed) ────────────────────────────────

class TestLoRA:
    def test_lora_config_defaults(self):
        from vla_hands.grafting.lora import LoRAConfig
        cfg = LoRAConfig()
        assert cfg.r == 8
        assert cfg.alpha == 16.0
        assert cfg.target_modules == "auto"

    def test_lora_parameter_count_import(self):
        from vla_hands.grafting.lora import lora_parameter_count
        # Should be importable regardless of whether peft is installed
        assert callable(lora_parameter_count)

    def test_apply_lora_without_peft_raises(self):
        """apply_lora raises ImportError with a helpful message if peft missing."""
        try:
            import peft  # noqa: F401
            pytest.skip("peft is installed; skipping ImportError test")
        except ImportError:
            from vla_hands.grafting.lora import apply_lora, LoRAConfig
            vlm = _FakeVLM()
            with pytest.raises((ImportError, RuntimeError)):
                apply_lora(vlm, LoRAConfig())

    def test_merge_lora_import(self):
        from vla_hands.grafting.lora import merge_lora
        assert callable(merge_lora)
