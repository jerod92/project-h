"""Tests for VisionBridge — universal vision skip connection wrapper."""

import pytest
import torch

from vla_hands.grafting.vision_bridge import VisionBridge
from vla_hands.appendages.joystick import JoystickAppendage
from vla_hands.appendages.dpad import DPadAppendage


HIDDEN_DIM = 64
VISION_DIM = 128
BATCH = 2


class TestVisionBridgeInit:
    def test_wraps_joystick(self):
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM)
        assert bridge.needs_vision_features is True
        assert bridge.hidden_dim == HIDDEN_DIM
        assert bridge.action_spec.name == "joystick"

    def test_wraps_dpad(self):
        dpad = DPadAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(dpad, vision_dim=VISION_DIM)
        assert bridge.action_spec.name == "dpad"
        assert bridge.action_spec.n_discrete == 5

    def test_gate_fusion(self):
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM, fusion="gate")
        assert bridge.gate is not None

    def test_add_fusion_no_gate(self):
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM, fusion="add")
        assert bridge.gate is None


class TestVisionBridgeForward:
    def test_joystick_with_vision(self):
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM)

        llm_feat = torch.randn(BATCH, HIDDEN_DIM)
        vis_feat = torch.randn(BATCH, VISION_DIM)
        out = bridge(llm_feat, vision_features=vis_feat)

        assert out.shape == (BATCH, 2)
        # Joystick outputs should be in [-1, 1] (Tanh)
        assert out.abs().max() <= 1.0

    def test_joystick_without_vision(self):
        """When vision_features is None, bridge is a passthrough."""
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM)

        llm_feat = torch.randn(BATCH, HIDDEN_DIM)
        out = bridge(llm_feat, vision_features=None)
        assert out.shape == (BATCH, 2)

    def test_dpad_with_vision(self):
        dpad = DPadAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(dpad, vision_dim=VISION_DIM)

        llm_feat = torch.randn(BATCH, HIDDEN_DIM)
        vis_feat = torch.randn(BATCH, VISION_DIM)
        out = bridge(llm_feat, vision_features=vis_feat)

        # DPad outputs logits over 5 classes
        assert out.shape == (BATCH, 5)

    def test_gate_fusion_forward(self):
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM, fusion="gate")

        llm_feat = torch.randn(BATCH, HIDDEN_DIM)
        vis_feat = torch.randn(BATCH, VISION_DIM)
        out = bridge(llm_feat, vision_features=vis_feat)
        assert out.shape == (BATCH, 2)


class TestVisionBridgeLoss:
    def test_action_loss_delegated(self):
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM)

        pred = torch.randn(BATCH, 2)
        target = torch.randn(BATCH, 2)
        loss = bridge.action_loss(pred, target)

        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_decode_delegated(self):
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM)

        action = torch.tensor([[0.5, -0.3]])
        decoded = bridge.decode(action)
        assert hasattr(decoded, "x")
        assert hasattr(decoded, "y")


class TestVisionBridgeGradients:
    def test_gradients_flow_through_bridge(self):
        """Vision projection weights should receive gradients during training."""
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM)

        llm_feat = torch.randn(BATCH, HIDDEN_DIM, requires_grad=True)
        vis_feat = torch.randn(BATCH, VISION_DIM, requires_grad=True)
        out = bridge(llm_feat, vision_features=vis_feat)
        loss = out.sum()
        loss.backward()

        # Vision projection should have gradients
        for p in bridge.vision_proj.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

        # LLM features should also have gradients
        assert llm_feat.grad is not None

    def test_small_init_near_identity(self):
        """With small init, bridge output should be close to no-bridge output."""
        joystick = JoystickAppendage(hidden_dim=HIDDEN_DIM)
        bridge = VisionBridge(joystick, vision_dim=VISION_DIM)

        llm_feat = torch.randn(1, HIDDEN_DIM)
        vis_feat = torch.randn(1, VISION_DIM)

        # Output without vision (passthrough)
        out_no_vision = bridge(llm_feat, vision_features=None)
        # Output with vision (should be close due to small init)
        out_with_vision = bridge(llm_feat, vision_features=vis_feat)

        diff = (out_no_vision - out_with_vision).abs().max().item()
        # Small init → small perturbation
        assert diff < 1.0, f"Bridge init changed output by {diff:.3f}, expected < 1.0"
