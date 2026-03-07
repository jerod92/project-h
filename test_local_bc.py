import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from vla_hands import (
    TargetNavEnvironment,
    VLAGraft, 
    GraftConfig, 
    JoystickAppendage,
    TrainingCurriculum, 
    CurriculumConfig, 
    QUICK_CURRICULUM
)
import time

def main():
    device = 'cpu'
    print(f"Testing on device: {device}")
    
    env = TargetNavEnvironment(width=224, height=224, max_steps=10)
    MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
    
    print(f"Loading processor and model {MODEL_ID}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vlm = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    print(f"Loaded in {time.time()-t0:.2f}s")
    
    text_cfg = getattr(vlm.config, 'text_config', None)
    hidden_dim = text_cfg.hidden_size if text_cfg is not None else vlm.config.hidden_size
    
    appendage = JoystickAppendage(hidden_dim=hidden_dim)
    graft = VLAGraft(
        vlm=vlm,
        appendage=appendage,
        config=GraftConfig(feature_extraction='last'),
    )
    
    print("Initializing Curriculum...")
    config = CurriculumConfig(
        bc_steps=2,          
        rl_steps=2,           
        device=device,
        save_dir='model_checkpoints/test_local',
        freezing_stages=QUICK_CURRICULUM,  
        eval_every=1,
        log_every=1,
        eval_episodes=1,
    )
    
    curriculum = TrainingCurriculum(
        graft=graft,
        processor=processor,
        environment=env,
        config=config,
    )
    
    print("Starting curriculum.run()...")
    t0 = time.time()
    curriculum.run()
    print(f"Finished successfully in {time.time()-t0:.2f}s!")

if __name__ == '__main__':
    main()
