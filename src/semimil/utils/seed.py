"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and (if available) PyTorch RNGs.

    Parameters
    ----------
    seed:
        The seed applied to every RNG.
    deterministic:
        If ``True``, also request deterministic cuDNN kernels. This makes runs
        bit-reproducible at some throughput cost; leave ``False`` for training
        speed and flip it on when auditing results.

    Returns
    -------
    int
        The seed that was applied (handy for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:  # pragma: no cover - torch optional at import time
        pass

    return seed
