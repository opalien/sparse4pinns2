import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Fixe la graine pour la reproductibilité des expériences.
    
    Args:
        seed (int): La graine à utiliser pour tous les générateurs de nombres aléatoires.
    """
    # Graine pour le module `random` de Python
    random.seed(seed)
    
    # Graine pour NumPy
    np.random.seed(seed)
    
    # Graine pour PyTorch pour le CPU
    torch.manual_seed(seed)
    
    # Graine pour PyTorch pour tous les GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # si vous utilisez plusieurs GPUs
        
        # Configuration pour rendre les opérations cuDNN déterministes
        # Attention : peut ralentir l'entraînement
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Certaines opérations (comme torch.bmm) peuvent encore être non-déterministes
        # sur les versions plus récentes de PyTorch. La ligne suivante peut aider.
        # Plus d'infos : https://pytorch.org/docs/stable/notes/randomness.html
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'