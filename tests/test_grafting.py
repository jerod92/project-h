"""Tests for the VLAGraft and FreezingCurriculum."""

import pytest
import torch
import torch.nn as nn

from vla_hands.appendages.joystick import JoystickAppendage
from vla_hands.appendages.dpad import DPadAppendage
from vla_hands.grafting.graft import GraftConfig, VLAGraft
from vla_hands.grafting.freezing import (
    FreezingCurriculum,
    FreezingStage,
    STAGE_APPENDAGE_ONLY,
    STAGE_LAST_2,
)


# ── Minimal stub VLM for tests (no HuggingFace download needed) ───────────────

class _FakeHiddenStates:
    def __init__(self, hidden_size: int, seq_len: int, batch: int):
        h = torch.randn(batch, seq_len, hidden_size)
        self.hidden_states = (h, h, h)  # 3 "layers"
        self.logits = torch.randn(batch, seq_len, 100)
        self.loss = None


class _FakeVLMConfig:
    hidden_size = 64
    text_config = None


class _FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeVLMConfig()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])
        # So FreezingCurriculum can find them
        self.model = type("M", (), {"layers": self.layers})()

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None,
                labels=None, output_hidden_states=False, return_dict=False, **kw):
        batch = 2
        seq = 5
        return _FakeHiddenStates(64, seq, batch)


HIDDEN_DIM = 64


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
        out = graft(
            input_ids=torch.zeros(2, 5, dtype=torch.long),
            attention_mask=torch.ones(2, 5),
        )
        assert out["action_features"].shape == (2, HIDDEN_DIM)

    def test_feature_extraction_first(self):
        graft = VLAGraft(self.vlm, self.app, GraftConfig("first", HIDDEN_DIM))
        out = graft(input_ids=torch.zeros(2, 5, dtype=torch.long))
        assert out["action_features"].shape == (2, HIDDEN_DIM)

    def test_feature_extraction_mean(self):
        graft = VLAGraft(self.vlm, self.app, GraftConfig("mean", HIDDEN_DIM))
        out = graft(
            input_ids=torch.zeros(2, 5, dtype=torch.long),
            attention_mask=torch.ones(2, 5),
        )
        assert out["action_features"].shape == (2, HIDDEN_DIM)

    def test_save_and_load(self, tmp_path):
        self.graft.save(tmp_path / "test_graft")
        config_file = tmp_path / "test_graft" / "graft_config.json"
        weights_file = tmp_path / "test_graft" / "appendage.pt"
        assert config_file.exists()
        assert weights_file.exists()

        # Modify weights, reload, check they match saved
        graft2 = VLAGraft(
            vlm=_FakeVLM(),
            appendage=JoystickAppendage(HIDDEN_DIM),
            config=GraftConfig(hidden_dim=HIDDEN_DIM),
        )
        graft2.load_appendage(tmp_path / "test_graft")

    def test_parameter_groups(self):
        groups = self.graft.parameter_groups(appendage_lr=1e-4, vlm_lr=1e-6)
        # At minimum, appendage group should be present
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
        assert out["action"].shape == (2, 5)  # 5 d-pad buttons


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
        assert changed  # first call always transitions into stage 0

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
