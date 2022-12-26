import numpy as np
import os
import torch
from typing import Optional, Tuple

from models.model import Model

def load_checkpoints(model: Model, checkpoint_path: str) \
    -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load the checkpoint file of a specific model in a specific folder and:
    * Set the state of the model according to the checkpoints;
    * Return the training and validation loss history along with the optimizer state.

    Parameters
    ----------
    model : Model
        The model for which the state is set according to the checkpoints.
    checkpoint_path : str
        The checkpoint path.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        Tuple containing in the given order the training loss history, the validation loss history and the optimizer state.
        (None, None, None) is returned if the checkpoint file is not present.
        - The training loss history is a list of pairs, containing the training loss1 (i.e. token importances loss) and the 
          training loss2 (i.e. answer generation loss) for each training step.
        - The validation loss history is a list of triples, containing the training step in which that validation step has 
          been performed, the validation loss1  (i.e. token importances loss) and the validation loss2 (i.e. answer 
          generation loss) for each validation step.
        - The optimizer state is a dictionary containing the state of the optmizer.
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        loss_history = checkpoint['loss_history']
        val_loss_history = checkpoint['val_loss_history']
        optimizer_state_dict = checkpoint['opt_state_dict']
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded saved files')
        return loss_history, val_loss_history, optimizer_state_dict 
    except:
        print('Unable to load saved files, default initialization')
        return None, None, None
    
def get_checkpoint_path(model_name: str, seed: int, use_history: bool) -> str:
    """Get the checkpoint file path.

    Parameters
    ----------
    model_name : str
        Name of the model for which the checkpoints path is returned ('distilroberta-base', 'prajjwal1/bert-tiny').
    seed : int
        The random seed used for reproducibility purposes for which the checkpoints path is returned (42, 2022, 1337).
    use_history : bool
        Whether to get the path of the checkpoints of the model using QaA discussion history or not. 

    Returns
    -------
    str
        The checkpoint path file name.
    """
    model_name = model_name.replace('/', '_')
    return os.path.join('weigths', 'PQH' if use_history else 'PQ', f'seed{seed}', f'{model_name}.pth')
