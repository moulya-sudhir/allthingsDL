import torch
from pathlib import Path

def save_model(model, target_dir, model_name):
    model_path = Path(target_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'model name should end with .pth or .pt'
    model_save_path = model_path / model_name
    print(f'[INFO] Saving model to: {model_save_path}')
    torch.save(obj=model.state_dict(), f=model_save_path)