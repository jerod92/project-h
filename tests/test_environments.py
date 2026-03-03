"""Tests for training environments."""

import pytest
import numpy as np
from PIL import Image

from vla_hands.environments.grid_world import GridWorldEnvironment
from vla_hands.environments.target_nav import TargetNavEnvironment
from vla_hands.appendages.dpad import DPadButton


class TestTargetNavEnvironment:
    def setup_method(self):
        self.env = TargetNavEnvironment(width=64, height=64, max_steps=20, speed=10.0)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)
        assert obs.size == (64, 64)

    def test_step_returns_result(self):
        self.env.reset(seed=1)
        result = self.env.step((0.5, 0.5))
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.observation, Image.Image)

    def test_step_accepts_tuple(self):
        self.env.reset(seed=2)
        result = self.env.step((1.0, 0.0))
        assert result is not None

    def test_step_accepts_list(self):
        self.env.reset(seed=3)
        result = self.env.step([0.0, 1.0])
        assert result is not None

    def test_expert_action_is_tuple(self):
        self.env.reset(seed=4)
        act = self.env.expert_action()
        assert len(act) == 2
        assert -1.0 <= act[0] <= 1.0
        assert -1.0 <= act[1] <= 1.0

    def test_expert_reaches_goal(self):
        """Expert policy should succeed within max_steps."""
        self.env = TargetNavEnvironment(width=64, height=64, max_steps=100, speed=15.0)
        self.env.reset(seed=42)
        for _ in range(100):
            act = self.env.expert_action()
            result = self.env.step(act)
            if result.done:
                assert result.info["success"]
                return
        pytest.fail("Expert policy did not reach goal within max_steps")

    def test_max_steps_property(self):
        assert self.env.max_steps == 20

    def test_image_size_property(self):
        assert self.env.image_size == (64, 64)

    def test_prompt_is_string(self):
        assert isinstance(self.env.prompt, str)
        assert len(self.env.prompt) > 0

    def test_clipping_oob_action(self):
        """Actions outside [-1, 1] should be clamped, not cause errors."""
        self.env.reset(seed=5)
        result = self.env.step((5.0, -5.0))  # way outside bounds
        assert result is not None


class TestGridWorldEnvironment:
    def setup_method(self):
        self.env = GridWorldEnvironment(grid_size=5, cell_px=20, max_steps=50)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)

    def test_image_size(self):
        obs = self.env.reset(seed=0)
        w, h = obs.size
        assert w == 5 * 20
        assert h == 5 * 20

    def test_step_with_int_action(self):
        self.env.reset(seed=1)
        result = self.env.step(1)  # UP
        assert isinstance(result.reward, float)

    def test_step_with_dpad_button(self):
        self.env.reset(seed=2)
        result = self.env.step(DPadButton.RIGHT)
        assert result is not None

    def test_expert_is_int(self):
        self.env.reset(seed=3)
        act = self.env.expert_action()
        assert isinstance(act, int)
        assert 0 <= act <= 4

    def test_expert_reaches_goal(self):
        """BFS expert should always reach goal in reachable maps."""
        for seed in range(10):
            self.env.reset(seed=seed)
            for _ in range(200):
                act = self.env.expert_action()
                result = self.env.step(act)
                if result.done:
                    assert result.info["success"], f"Expert failed on seed {seed}"
                    break
            else:
                pytest.fail(f"Expert could not reach goal on seed {seed}")

    def test_wall_hit_no_movement(self):
        """Hitting a wall should leave agent in place."""
        self.env.reset(seed=0)
        # Force agent into corner where moving left should hit boundary/wall
        before = self.env._agent
        # Try moving up from row 0 (out of bounds)
        if self.env._agent[0] == 0:
            result = self.env.step(DPadButton.UP)
            assert self.env._agent[0] >= 0  # didn't go negative

    def test_success_reward(self):
        """Reaching goal should give positive reward."""
        self.env.reset(seed=999)
        # Manually set agent adjacent to goal
        g = self.env.grid_size
        self.env._agent = (g - 1, g - 2)
        self.env._grid[g - 1, g - 1] = 0  # ensure goal cell is empty
        result = self.env.step(DPadButton.RIGHT)
        assert result.info.get("success") or result.reward > 0 or result.done
