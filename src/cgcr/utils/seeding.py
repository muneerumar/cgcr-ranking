"""Seed Python, NumPy, and optional PyTorch RNGs; implemented in ."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Reset RNGs and deterministic cuDNN settings when torch is installed.

    Reproducibility is scoped to the same software and device environment;
    this does not promise identical results across platforms or all kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
