import os
import torch

from models.model import Model

def load_checkpoints(model: Model, folder_name: str, model_name: str):
    try:
        checkpoint = torch.load(os.path.join(folder_name, f"{model_name.replace('/','_')}.pth"))
        loss_history = checkpoint['loss_history']
        val_loss_history = checkpoint['val_loss_history']
        opt_state_dict = checkpoint['opt_state_dict']
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded saved files')
        return loss_history, val_loss_history, opt_state_dict 
    except:
        print('Unable to load saved files, default initialization')
        return None, None, None
    
def get_checkpoint_folder(seed: int, use_history: bool):
    folder = 'weigths'
    if use_history:
        return os.path.join(f'{folder}/PQH/seed{seed}')
    return os.path.join(f'{folder}/PQ/seed{seed}')