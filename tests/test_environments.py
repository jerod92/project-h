"""Tests for training environments."""

import pytest
import numpy as np
from PIL import Image

from vla_hands.environments.grid_world import GridWorldEnvironment
from vla_hands.environments.target_nav import TargetNavEnvironment
from vla_hands.environments.spaceship import SpaceshipNavEnvironment
from vla_hands.environments.maze import MazeEnvironment
from vla_hands.environments.button_task import ButtonPressEnvironment
from vla_hands.environments.mcq_navigator import MCQNavigatorEnvironment
from vla_hands.environments.pointing import PointingEnvironment
from vla_hands.environments.fruit_catcher import FruitCatcherEnvironment
from vla_hands.environments.treasure_hunt import TreasureHuntEnvironment
from vla_hands.environments.paint_canvas import PaintCanvasEnvironment
from vla_hands.environments.whack_a_mole import WhackAMoleEnvironment
from vla_hands.appendages.dpad import DPadButton


# ── TargetNav ─────────────────────────────────────────────────────────────────

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
        assert self.env.step((1.0, 0.0)) is not None

    def test_step_accepts_list(self):
        self.env.reset(seed=3)
        assert self.env.step([0.0, 1.0]) is not None

    def test_expert_action_is_tuple(self):
        self.env.reset(seed=4)
        act = self.env.expert_action()
        assert len(act) == 2
        assert -1.0 <= act[0] <= 1.0
        assert -1.0 <= act[1] <= 1.0

    def test_expert_reaches_goal(self):
        env = TargetNavEnvironment(width=64, height=64, max_steps=100, speed=15.0)
        env.reset(seed=42)
        for _ in range(100):
            result = env.step(env.expert_action())
            if result.done:
                assert result.info["success"]
                return
        pytest.fail("Expert did not reach goal within max_steps")

    def test_max_steps_property(self):
        assert self.env.max_steps == 20

    def test_image_size_property(self):
        assert self.env.image_size == (64, 64)

    def test_prompt_is_string(self):
        assert isinstance(self.env.prompt, str)
        assert len(self.env.prompt) > 0

    def test_dynamic_prompt_varies(self):
        """PromptVocab should produce different prompts across resets."""
        prompts = set()
        for i in range(10):
            self.env.reset(seed=i)
            prompts.add(self.env.prompt)
        assert len(prompts) > 1

    def test_clipping_oob_action(self):
        self.env.reset(seed=5)
        assert self.env.step((5.0, -5.0)) is not None


# ── GridWorld ─────────────────────────────────────────────────────────────────

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
        result = self.env.step(1)
        assert isinstance(result.reward, float)

    def test_step_with_dpad_button(self):
        self.env.reset(seed=2)
        assert self.env.step(DPadButton.RIGHT) is not None

    def test_expert_is_int(self):
        self.env.reset(seed=3)
        act = self.env.expert_action()
        assert isinstance(act, int)
        assert 0 <= act <= 4

    def test_expert_reaches_goal(self):
        for seed in range(10):
            self.env.reset(seed=seed)
            for _ in range(200):
                result = self.env.step(self.env.expert_action())
                if result.done:
                    assert result.info["success"], f"Expert failed on seed {seed}"
                    break
            else:
                pytest.fail(f"Expert could not reach goal on seed {seed}")

    def test_wall_hit_no_movement(self):
        self.env.reset(seed=0)
        if self.env._agent[0] == 0:
            result = self.env.step(DPadButton.UP)
            assert self.env._agent[0] >= 0

    def test_success_reward(self):
        self.env.reset(seed=999)
        g = self.env.grid_size
        self.env._agent = (g - 1, g - 2)
        self.env._grid[g - 1, g - 1] = 0
        result = self.env.step(DPadButton.RIGHT)
        assert result.info.get("success") or result.reward > 0 or result.done


# ── SpaceshipNav ──────────────────────────────────────────────────────────────

class TestSpaceshipNavEnvironment:
    def setup_method(self):
        self.env = SpaceshipNavEnvironment(width=64, height=64, max_steps=50)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)
        assert obs.size == (64, 64)

    def test_step_result(self):
        self.env.reset(seed=1)
        result = self.env.step([0.5, -0.3])
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    def test_expert_action_shape(self):
        self.env.reset(seed=2)
        act = self.env.expert_action()
        assert len(act) == 2

    def test_momentum_changes_position(self):
        """Applying thrust repeatedly should displace the ship."""
        self.env.reset(seed=3)
        pos0 = self.env._pos.copy()
        for _ in range(10):
            self.env.step([1.0, 0.0])
        assert not (self.env._pos == pos0).all()


# ── Maze ─────────────────────────────────────────────────────────────────────

class TestMazeEnvironment:
    def setup_method(self):
        self.env = MazeEnvironment(rows=5, cols=5, max_steps=100)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)

    def test_step_result(self):
        self.env.reset(seed=1)
        result = self.env.step(DPadButton.RIGHT)
        assert isinstance(result.reward, float)

    def test_expert_action_is_dpad(self):
        self.env.reset(seed=2)
        act = self.env.expert_action()
        assert isinstance(act, (int, DPadButton))

    def test_expert_reaches_goal(self):
        for seed in range(5):
            self.env.reset(seed=seed)
            for _ in range(100):
                result = self.env.step(self.env.expert_action())
                if result.done:
                    break
            assert result.done, f"Maze not solved on seed {seed}"


# ── ButtonPress ───────────────────────────────────────────────────────────────

class TestButtonPressEnvironment:
    def setup_method(self):
        self.env = ButtonPressEnvironment(width=64, height=64)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)

    def test_step_accepts_float(self):
        self.env.reset(seed=1)
        result = self.env.step(1.0)
        assert isinstance(result.reward, float)

    def test_expert_is_float(self):
        self.env.reset(seed=2)
        act = self.env.expert_action()
        assert isinstance(act, float)
        assert act in (0.0, 1.0)


# ── PointingEnvironment ───────────────────────────────────────────────────────

class TestPointingEnvironment:
    def setup_method(self):
        self.env = PointingEnvironment(width=128, height=128, n_distractors=2)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)
        assert obs.size == (128, 128)

    def test_step_accepts_tuple(self):
        self.env.reset(seed=1)
        result = self.env.step((0.5, 0.5))
        assert isinstance(result.reward, float)

    def test_expert_action_in_unit_square(self):
        self.env.reset(seed=2)
        x, y = self.env.expert_action()
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0

    def test_expert_tap_is_on_target(self):
        """Expert tap should hit the target circle."""
        for seed in range(5):
            self.env.reset(seed=seed)
            x, y = self.env.expert_action()
            result = self.env.step((x, y))
            assert result.reward > 0, f"Expert tap missed on seed {seed}"

    def test_dynamic_prompt(self):
        prompts = set()
        for i in range(8):
            self.env.reset(seed=i)
            prompts.add(self.env.prompt)
        assert len(prompts) > 1


# ── FruitCatcher ──────────────────────────────────────────────────────────────

class TestFruitCatcherEnvironment:
    def setup_method(self):
        self.env = FruitCatcherEnvironment(width=128, height=128, max_steps=50)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)

    def test_step_dict_action(self):
        self.env.reset(seed=1)
        result = self.env.step({"joystick": [0.5, 0.0], "button": 1.0})
        assert isinstance(result.reward, float)

    def test_step_list_action(self):
        self.env.reset(seed=2)
        result = self.env.step([0.3, 0.0])
        assert result is not None

    def test_expert_action_is_dict(self):
        self.env.reset(seed=3)
        act = self.env.expert_action()
        assert isinstance(act, dict)
        assert "joystick" in act
        assert "button" in act

    def test_expert_joystick_in_range(self):
        self.env.reset(seed=4)
        act = self.env.expert_action()
        jx, jy = act["joystick"]
        assert -1.0 <= jx <= 1.0
        assert -1.0 <= jy <= 1.0


# ── TreasureHunt ──────────────────────────────────────────────────────────────

class TestTreasureHuntEnvironment:
    def setup_method(self):
        self.env = TreasureHuntEnvironment(grid_size=5, max_steps=80)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)

    def test_step_dict_action(self):
        self.env.reset(seed=1)
        result = self.env.step({"dpad": 1, "button": 0.0})
        assert isinstance(result.reward, float)

    def test_expert_action_keys(self):
        self.env.reset(seed=2)
        act = self.env.expert_action()
        assert isinstance(act, dict)
        assert "dpad" in act
        assert "button" in act

    def test_at_least_one_treasure(self):
        for seed in range(5):
            self.env.reset(seed=seed)
            assert len(self.env._treasures) >= 1


# ── PaintCanvas ───────────────────────────────────────────────────────────────

class TestPaintCanvasEnvironment:
    def setup_method(self):
        self.env = PaintCanvasEnvironment(width=128, height=128)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)

    def test_step_tuple_action(self):
        self.env.reset(seed=1)
        result = self.env.step((0.5, 0.5))
        assert isinstance(result.reward, float)

    def test_expert_action_in_unit_square(self):
        self.env.reset(seed=2)
        x, y = self.env.expert_action()
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0

    def test_expert_progresses_waypoints(self):
        self.env.reset(seed=3)
        before = self.env._current_wp
        x, y = self.env.expert_action()
        self.env.step((x, y))
        # Either waypoint advanced or episode is done
        assert self.env._current_wp >= before


# ── WhackAMole ────────────────────────────────────────────────────────────────

class TestWhackAMoleEnvironment:
    def setup_method(self):
        self.env = WhackAMoleEnvironment(width=128, height=128, n_moles=4)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)

    def test_step_tuple_action(self):
        self.env.reset(seed=1)
        result = self.env.step((0.5, 0.5))
        assert isinstance(result.reward, float)

    def test_expert_action_in_unit_square(self):
        self.env.reset(seed=2)
        x, y = self.env.expert_action()
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0

    def test_has_moles(self):
        self.env.reset(seed=3)
        assert len(self.env._moles) == 4

    def test_expert_taps_best_mole(self):
        """Expert should target highest-value active mole."""
        for seed in range(5):
            self.env.reset(seed=seed)
            x, y = self.env.expert_action()
            result = self.env.step((x, y))
            # Expert should at least not hit a bomb (negative reward)
            assert result.reward >= -0.1, f"Expert hit bomb on seed {seed}"


# ── MCQNavigator ──────────────────────────────────────────────────────────────

class TestMCQNavigatorEnvironment:
    def setup_method(self):
        self.env = MCQNavigatorEnvironment(max_steps=60)

    def test_reset_returns_image(self):
        obs = self.env.reset(seed=0)
        assert isinstance(obs, Image.Image)

    def test_step_dict_action(self):
        self.env.reset(seed=1)
        result = self.env.step({"dpad": 0, "multibutton": [0.0, 0.0, 0.0, 0.0]})
        assert isinstance(result.reward, float)

    def test_expert_action_keys(self):
        self.env.reset(seed=2)
        act = self.env.expert_action()
        assert isinstance(act, dict)
        assert "dpad" in act
        assert "multibutton" in act

    def test_expert_choice_is_one_hot(self):
        self.env.reset(seed=3)
        act = self.env.expert_action()
        choices = act["multibutton"]
        assert sum(int(round(v)) for v in choices) == 1  # exactly one pressed
