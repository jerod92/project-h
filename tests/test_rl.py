import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from PIL import Image

from vla_hands.training.trainer import RLTrainer, TrainerConfig
from vla_hands.environments.base import BaseEnvironment, EnvStepResult
from vla_hands.grafting.graft import VLAGraft, GraftConfig
from vla_hands.appendages.dpad import DPadAppendage

class MockEnv(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self._step = 0
        self._prompt = "test"
        
    @property
    def max_steps(self) -> int:
        return 10
        
    @property
    def image_size(self) -> tuple[int, int]:
        return (32, 32)
        
    @property
    def prompt(self) -> str:
        return self._prompt
        
    def reset(self, seed=None):
        self._step = 0
        return Image.new("RGB", (32, 32))
        
    def step(self, action):
        self._step += 1
        return EnvStepResult(
            observation=Image.new("RGB", (32, 32)),
            reward=1.0 if action == 1 else 0.0,
            done=(self._step >= self.max_steps),
            info={"success": True} if self._step >= self.max_steps else {}
        )
        
    def expert_action(self):
        return 1

class MockProcessor:
    def __call__(self, images, text, return_tensors, padding=True):
        return {"pixel_values": torch.zeros(1, 3, 32, 32), "input_ids": torch.zeros(1, 5, dtype=torch.long)}

class MockVLMConfig:
    def __init__(self):
        self.hidden_dim = 16

class MockVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockVLMConfig()
        
    def forward(self, *args, **kwargs):
        class Output:
            hidden_states = [torch.ones(1, 5, 16)]
            logits = torch.ones(1, 5, 10)
        return Output()

def test_rl_trainer_runs():
    vlm = MockVLM()
    appendage = DPadAppendage(hidden_dim=16)
    graft_config = GraftConfig(hidden_dim=16)
    graft = VLAGraft(vlm=vlm, appendage=appendage, config=graft_config)
    
    # We must properly configure the graft config
    # Wait, VLAGraft has a config. Let's provide a basic one if needed.
    
    config = TrainerConfig(
        rl_steps=2,
        rl_episodes_per_update=2,
        rl_max_steps_per_episode=5
    )
    
    trainer = RLTrainer(
        graft=graft,
        processor=MockProcessor(),
        environment=MockEnv(),
        config=config,
        device="cpu"
    )
    
    metrics = trainer.train()
    assert len(metrics) > 0
