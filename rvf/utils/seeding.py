"""Reproducible seeding across random, numpy and torch (if present)."""
from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed all RNGs we might touch. torch is optional so the data layer can be
    exercised without it installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
