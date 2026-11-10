from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class ExpandingWindowLoader(Dataset):
    """
    A PyTorch Dataset designed explicitly for expanding-prefix time-series forecasting.

    This loader takes multiple independent temporal sequences and slices them into progressively larger
    temporal windows (e.g., [0-20], [0-40], [0-60] days).
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
        Initializes and pre-computes all expanding sliding windows for the entire dataset.

        :param List[np.ndarray] x_list: List of 2D arrays containing exogenous input sequences (one array per state).
        :param List[np.ndarray] y_list: List of 2D arrays containing ground-truth target trajectories.
        :param List[np.ndarray] t_list: List of 2D arrays representing unscaled time vectors.
        :param List[np.ndarray] aux_list: List of 2D arrays containing ODE pre-calibrated auxiliary curves.
        :param int min_days: The minimum number of historical days required before predicting the future.
        :param int max_days: The maximum sequence length in the dataset. Used to standardize tensor sizes via padding.
        :param int step: The stride/interval at which the window expands (e.g., expanding by 10 days at a time).
        :param str device: The target computation device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.samples = []

        # Iterate over all independently simulated states (e.g., 20 different regions or parameterizations)
        for x, y, t, aux in zip(x_list, y_list, t_list, aux_list):

            # Slice each full sequence into progressively larger historical windows
            for current_days in range(min_days, max_days + 1, step):
                # Padding the exogenous inputs
                # Shorter sequences are padded with 0.0 up to max_days to ensure uniform tensor shapes for batching
                x_padded = torch.zeros((max_days, x.shape[1]), dtype=torch.float32, device=self.device)
                x_padded[:current_days] = torch.tensor(x[:current_days], dtype=torch.float32, device=self.device)

                # Generating the Attention Mask
                # A float mask where 1.0 represents valid real data and 0.0 represents the padded future.
                # (Note: PyTorch MultiHeadAttention requires boolean masks where True = padding, so this mask
                # will be inverted inside the Dataset __getitem__ or the Runner logic before being fed to MHA).
                mask = torch.zeros(max_days, dtype=torch.float32, device=self.device)
                mask[:current_days] = 1.0

                # 3. Store the pre-computed window as a dictionary ready for the DataLoader
                self.samples.append({
                    'X': x_padded,
                    'mask': mask,
                    'y': torch.tensor(y, dtype=torch.float32, device=self.device),
                    't': torch.tensor(t, dtype=torch.float32, device=self.device),
                    'aux_targets': torch.tensor(aux, dtype=torch.float32, device=self.device)
                })

    def __len__(self) -> int:
        """
        Returns the total number of expanding windows generated across all states.

        :return int: Total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a specific windowed dictionary by index.

        :param int index: The batch index requested by the PyTorch DataLoader.
        :return Dict[str, torch.Tensor]: A dictionary containing the padded tensors and masks.
        """
        return self.samples[index]
