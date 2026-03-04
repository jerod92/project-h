"""Tests for action appendages."""

import pytest
import torch

from vla_hands.appendages.dpad import DPadAppendage, DPadButton, DPAD_DELTA
from vla_hands.appendages.joystick import JoystickAppendage, JoystickAction
from vla_hands.appendages.button import ButtonAppendage, ButtonState, MultiButtonAppendage, MultiButtonState
from vla_hands.appendages.touchscreen import TouchscreenAppendage


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


class TestButtonAppendage:
    def setup_method(self):
        self.app = ButtonAppendage(hidden_dim=HIDDEN_DIM)

    def test_output_shape(self):
        h = torch.randn(4, HIDDEN_DIM)
        out = self.app(h)
        assert out.shape == (4, 1)

    def test_output_range(self):
        h = torch.randn(8, HIDDEN_DIM) * 100
        out = self.app(h)
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_loss_scalar(self):
        h = torch.randn(4, HIDDEN_DIM)
        pred = self.app(h)
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = self.app.action_loss(pred, target)
        assert loss.shape == ()

    def test_loss_2d_target(self):
        h = torch.randn(4, HIDDEN_DIM)
        pred = self.app(h)
        target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        loss = self.app.action_loss(pred, target)
        assert loss.shape == ()

    def test_is_pressed(self):
        t = torch.tensor([[0.8], [0.3], [0.6]])
        pressed = self.app.is_pressed(t)
        assert pressed.tolist() == [True, False, True]

    def test_decode(self):
        t = torch.tensor([[0.75]])
        state = self.app.decode(t)
        assert isinstance(state, ButtonState)
        assert state.pressed is True
        assert abs(state.confidence - 0.75) < 1e-5

    def test_decode_not_pressed(self):
        t = torch.tensor([[0.2]])
        state = self.app.decode(t)
        assert state.pressed is False

    def test_action_spec(self):
        spec = self.app.action_spec
        assert spec.dtype == "continuous"
        assert spec.shape == (1,)


class TestMultiButtonAppendage:
    def setup_method(self):
        self.app = MultiButtonAppendage(hidden_dim=HIDDEN_DIM, n_buttons=4, labels=["A", "B", "C", "D"])

    def test_output_shape(self):
        h = torch.randn(4, HIDDEN_DIM)
        out = self.app(h)
        assert out.shape == (4, 4)

    def test_output_range(self):
        h = torch.randn(8, HIDDEN_DIM) * 100
        out = self.app(h)
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_loss_scalar(self):
        h = torch.randn(4, HIDDEN_DIM)
        pred = self.app(h)
        target = torch.zeros(4, 4)
        target[0, 1] = 1.0
        target[1, 3] = 1.0
        loss = self.app.action_loss(pred, target)
        assert loss.shape == ()

    def test_argmax_button(self):
        h = torch.randn(4, HIDDEN_DIM)
        out = self.app(h)
        idx = self.app.argmax_button(out)
        assert idx.shape == (4,)
        assert idx.min() >= 0
        assert idx.max() <= 3

    def test_pressed_buttons(self):
        t = torch.tensor([[0.8, 0.2, 0.7, 0.1]])
        result = self.app.pressed_buttons(t)
        assert result == [[0, 2]]

    def test_decode(self):
        t = torch.tensor([[0.9, 0.1, 0.6, 0.4]])
        state = self.app.decode(t)
        assert isinstance(state, MultiButtonState)
        assert state.pressed == [True, False, True, False]
        assert state.any_pressed()
        assert state.pressed_indices() == [0, 2]

    def test_labels_stored(self):
        assert self.app.labels == ["A", "B", "C", "D"]

    def test_default_labels(self):
        app = MultiButtonAppendage(HIDDEN_DIM, n_buttons=3)
        assert app.labels == ["0", "1", "2"]

    def test_action_spec(self):
        spec = self.app.action_spec
        assert spec.shape == (4,)


class TestTouchscreenAppendage:
    def test_output_shape(self):
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM)
        h = torch.randn(4, HIDDEN_DIM)
        out = app(h)
        assert out.shape == (4, 2)

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM)
        h = torch.randn(8, HIDDEN_DIM) * 100
        out = app(h)
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_output_shape_with_vision(self):
        vision_dim = 32
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM, vision_dim=vision_dim)
        h = torch.randn(4, HIDDEN_DIM)
        vis = torch.randn(4, vision_dim)
        out = app(h, vision_features=vis)
        assert out.shape == (4, 2)

    def test_no_vision_proj_without_dim(self):
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM, vision_dim=None)
        assert app.vision_proj is None

    def test_vision_proj_with_dim(self):
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM, vision_dim=32)
        assert app.vision_proj is not None

    def test_loss_scalar(self):
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM)
        h = torch.randn(4, HIDDEN_DIM)
        pred = app(h)
        target = torch.rand(4, 2)
        loss = app.action_loss(pred, target)
        assert loss.shape == ()

    def test_decode(self):
        from vla_hands.appendages.touchscreen import TouchPoint
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM)
        t = torch.tensor([[0.3, 0.7]])
        tp = app.decode(t)
        assert isinstance(tp, TouchPoint)
        assert abs(tp.x - 0.3) < 1e-4
        assert abs(tp.y - 0.7) < 1e-4

    def test_no_grad_warning(self):
        """decode() must not warn about grad on non-leaf tensor."""
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM)
        h = torch.randn(1, HIDDEN_DIM)
        out = app(h)  # has requires_grad via parameters
        tp = app.decode(out)  # should not raise/warn
        assert isinstance(tp.x, float)

    def test_action_spec(self):
        app = TouchscreenAppendage(hidden_dim=HIDDEN_DIM)
        spec = app.action_spec
        assert spec.shape == (2,)
        assert spec.low == 0.0
        assert spec.high == 1.0
