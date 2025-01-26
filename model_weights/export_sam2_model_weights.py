import torch
import os

root_dir = "SAM2_finetuned"

for fold in os.listdir(root_dir):
    fold_path = os.path.join(root_dir, fold)
    ckpt_path = os.path.join(fold_path, "checkpoints", "checkpoint.pt")
    model_weights_path = os.path.join(fold_path, "checkpoint.pt")
    
    if os.path.isfile(ckpt_path):
        print(f"Processing: {fold}")
        
        file_finetuned = torch.load(ckpt_path)
        model_only_checkpoint = {'model': file_finetuned['model']}
        torch.save(model_only_checkpoint, model_weights_path)
    
        print(f"Successfully converted {fold} to {model_weights_path}")  
        