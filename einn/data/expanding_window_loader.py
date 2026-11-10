from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class ExpandingWindowLoader(Dataset):
    """
    Data Loader designed explicitly for expanding-prefix forecasting strategies.

    It takes multiple time-series sequences (e.g., from different synthetic states or real regions)
    and slices them into progressively larger temporal windows (from min_days up to max_days).
    It automatically applies zero-padding to shorter windows and generates standard attention
    masks to ensure PyTorch compatibility (especially for Multi-Head Attention layers).
    """

    def __init__(
            self,
            x_list: List[np.ndarray],
            y_list: List[np.ndarray],
            t_list: List[np.ndarray],
            aux_list: List[np.ndarray],
            min_days: int = 10,
            max_days: int = 200,
            step: int = 10,
            device: str = 'cpu'
    ):
        """
        Initializes and pre-computes all expanding windows for the dataset.

        :param List[np.ndarray] x_list: List of exogenous input sequences (one per state).
        :param List[np.ndarray] y_list: List of ground-truth target sequences.
        :param List[np.ndarray] t_list: List of unscaled time vectors.
        :param List[np.ndarray] aux_list: List of ODE pre-calibrated auxiliary curves.
        :param int min_days: Minimum historical days needed before the first prediction window.
        :param int max_days: Maximum days available in the dataset (used for global padding).
        :param int step: The interval at which the window expands (e.g., every 10 days).
        :param str device: Computation device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.samples = []

        # Iterate over all independently simulated states (e.g., 20 different parameterizations)
        for x, y, t, aux in zip(x_list, y_list, t_list, aux_list):

            # Slice each sequence into progressively larger windows
            for current_days in range(min_days, max_days + 1, step):
                # Zero-padding the exogenous inputs up to max_days to ensure batch consistency
                x_padded = torch.zeros((max_days, x.shape[1]), dtype=torch.float32, device=self.device)
                x_padded[:current_days] = torch.tensor(x[:current_days], dtype=torch.float32, device=self.device)

                # Generating attention masks. PyTorch MHA (key_padding_mask) expects True for padded elements.
                # Here, we generate a float mask where 0.0 is padding and 1.0 is valid data.
                # Note: The Runner/Trainer should invert this to Boolean (False=valid, True=padding) if used in MHA.
                mask = torch.zeros(max_days, dtype=torch.float32, device=self.device)
                mask[:current_days] = 1.0

                self.samples.append({
                    'X': x_padded,
                    'mask': mask,
                    'y': torch.tensor(y, dtype=torch.float32, device=self.device),
                    't': torch.tensor(t, dtype=torch.float32, device=self.device),
                    'aux_targets': torch.tensor(aux, dtype=torch.float32, device=self.device)
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]
