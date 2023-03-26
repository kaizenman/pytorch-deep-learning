import torch 
from pathlib import Path

def save_model(model, dir, model_name):
  target_dir = Path(dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  model_save_path = Path(target_dir / model_name)
  print(f'Saving model to {model_save_path}...')
  torch.save(obj=model.state_dict(), f=model_save_path)
  print(f'Saved.')