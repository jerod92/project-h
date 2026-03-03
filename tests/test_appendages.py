"""Tests for action appendages."""

import pytest
import torch

from vla_hands.appendages.dpad import DPadAppendage, DPadButton, DPAD_DELTA
from vla_hands.appendages.joystick import JoystickAppendage, JoystickAction


HIDDEN_DIM = 64


class TestJoystickAppendage:
    def setup_method(self):
        self.app = JoystickAppendage(hidden_dim=HIDDEN_DIM)

    def test_output_shape(self):
        h = torch.randn(4, HIDDEN_DIM)
        out = self.app(h)
        assert out.shape == (4, 2)

    def test_output_range(self):
        h = torch.randn(8, HIDDEN_DIM) * 100  # large input
        out = self.app(h)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_loss_scalar(self):
        h = torch.randn(4, HIDDEN_DIM)
        pred = self.app(h)
        target = torch.zeros(4, 2)
        loss = self.app.action_loss(pred, target)
        assert loss.shape == ()

    def test_loss_with_weights(self):
        h = torch.randn(4, HIDDEN_DIM)
        pred = self.app(h)
        target = torch.zeros(4, 2)
        weights = torch.tensor([1.0, 0.5, 0.5, 1.0])
        loss = self.app.action_loss(pred, target, weights=weights)
        assert loss.shape == ()

    def test_decode(self):
        t = torch.tensor([[0.5, -0.3]])
        action = self.app.decode(t)
        assert isinstance(action, JoystickAction)
        assert abs(action.x - 0.5) < 1e-5
        assert abs(action.y - (-0.3)) < 1e-5

    def test_action_spec(self):
        spec = self.app.action_spec
        assert spec.dtype == "continuous"
        assert spec.shape == (2,)
        assert spec.low == -1.0
        assert spec.high == 1.0


class TestDPadAppendage:
    def setup_method(self):
        self.app = DPadAppendage(hidden_dim=HIDDEN_DIM)

    def test_output_shape(self):
        h = torch.randn(4, HIDDEN_DIM)
        out = self.app(h)
        assert out.shape == (4, 5)

    def test_argmax_range(self):
        h = torch.randn(4, HIDDEN_DIM)
        logits = self.app(h)
        actions = self.app.argmax(logits)
        assert actions.shape == (4,)
        assert actions.min() >= 0
        assert actions.max() <= 4

    def test_sample(self):
        h = torch.randn(4, HIDDEN_DIM)
        logits = self.app(h)
        actions = self.app.sample(logits)
        assert actions.shape == (4,)
        assert actions.min() >= 0
        assert actions.max() <= 4

    def test_loss_scalar(self):
        h = torch.randn(4, HIDDEN_DIM)
        pred = self.app(h)
        target = torch.zeros(4, dtype=torch.long)
        loss = self.app.action_loss(pred, target)
        assert loss.shape == ()

    def test_decode(self):
        t = torch.tensor([[0.1, 0.2, 0.05, 0.3, 0.35]])  # RIGHT has max
        btn = self.app.decode(t)
        assert btn == DPadButton.RIGHT

    def test_action_spec(self):
        spec = self.app.action_spec
        assert spec.dtype == "discrete"
        assert spec.n_discrete == 5

    def test_dpad_delta_completeness(self):
        for btn in DPadButton:
            assert btn in DPAD_DELTA
