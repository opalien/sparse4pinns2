import random
import numpy as np
import torch

import os

def set_seed(seed: int):

    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")