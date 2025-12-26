"""
Utility functions for Quantum Multilayer Perceptron (QMLP) training.
"""


import numpy as np
import random
import os


def set_reproducibility(seed=42):
    """
    Set random seeds for full reproducibility:
    - Python
    - NumPy
    - Optuna
    - PyTorch (CPU + CUDA)
    """
    import os
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # -------- PyTorch (Optuna often interacts with it) --------
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # -------- Optuna --------
    try:
        import optuna
        optuna.samplers.RandomSampler(seed=seed)
        optuna.samplers.TPESampler(seed=seed)
    except ImportError:
        pass
