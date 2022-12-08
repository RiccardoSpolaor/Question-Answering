import numpy as np
import os
import random 
import torch

def set_random_seed(random_seed: int = 42) -> None:
    """Set the random seed for reproducibility. The seed is set for the random library, the numpy library and the pytorch 
    library. Moreover the environment variable `TF_DETERMINISTIC_OPS` is set as "1".

    Parameters
    ----------
    random_seed : int, optional
        The random seed to use for reproducibility (default 42).
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['TF_DETERMINISTIC_OPS'] = '1'