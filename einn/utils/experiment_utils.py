import os
import random

import numpy as np
import torch


class ExperimentUtils:
    """
    Utility class for global experiment settings and deterministic behavior.
    """
    @staticmethod
    def seed_everything(seed: int = 42):
        """
        Globally fixes all random number generators to ensure 100% reproducibility.

        :param int seed: The seed value to use across all libraries.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Enforce CuDNN determinism (critical for RNNs/LSTMs)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Global seed set to {seed}.")
