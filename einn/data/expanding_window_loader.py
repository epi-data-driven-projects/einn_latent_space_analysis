from typing import Dict, List

import numpy as np
import torch

from einn.data.einn_dataset import EINNDataset


class ExpandingWindowLoader(EINNDataset):
    """
    PyTorch Dataset implementing the expanding-window scheme from thesis Section 2.3.

    For each simulated trajectory of length max_days, generates nested observation prefixes
    of lengths min_days, min_days + step, ..., up to max_days. Unobserved future entries in
    the input tensor X are zero-padded to max_days so all prefixes can be batched uniformly,
    while full-horizon targets (y, t, aux_targets) remain unmasked for loss computation.
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
        Pre-compute and cache all expanding-prefix samples across all trajectories.

        :param List[np.ndarray] x_list: Standardized input sequences, each of shape (T, d_x).
        :param List[np.ndarray] y_list: Standardized target trajectories, each of shape (T, 1).
        :param List[np.ndarray] t_list: Unscaled time index grids, each of shape (T, 1).
        :param List[np.ndarray] aux_list: Standardized auxiliary compartment states, each of shape (T, d_s).
        :param int min_days: Shortest observation prefix length included in training.
        :param int max_days: Full trajectory horizon T used as the padded sequence length.
        :param int step: Increment between consecutive expanding-window prefix lengths.
        :param str device: Target PyTorch device string (e.g. 'cpu' or 'cuda:0').
        """
        self.device = torch.device(device)
        self.samples: List[Dict[str, torch.Tensor]] = []

        # Build nested observation prefixes for every independently simulated trajectory.
        for x, y, t, aux in zip(x_list, y_list, t_list, aux_list):
            for current_days in range(min_days, max_days + 1, step):
                # Copy the observed prefix [0, current_days) and zero-pad the unobserved future to (max_days, d_x).
                x_padded = torch.zeros((max_days, x.shape[1]), dtype=torch.float32, device=self.device)
                x_padded[:current_days] = torch.tensor(x[:current_days], dtype=torch.float32, device=self.device)

                # Build padding mask of shape (max_days,): 0.0 marks valid observed steps, 1.0 marks padded future steps
                mask = torch.ones(max_days, dtype=torch.float32, device=self.device)
                mask[:current_days] = 0.0

                # Store padded prefix inputs together with full-horizon targets (y, t, aux_targets).
                self.samples.append({
                    'X': x_padded,                                                      # Shape: (max_days, d_x)
                    'mask': mask,                                                       # Shape: (max_days,)
                    'y': torch.tensor(y, dtype=torch.float32, device=self.device),      # Shape: (T, 1)
                    't': torch.tensor(t, dtype=torch.float32, device=self.device),      # Shape: (T, 1)
                    'aux_targets': torch.tensor(aux, dtype=torch.float32, device=self.device)  # Shape: (T, d_s)
                })

    def __len__(self) -> int:
        """
        Return the total number of expanding-window samples across all trajectories.

        :return int: Number of pre-computed (trajectory, prefix_length) pairs.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single pre-computed expanding-window sample by index.

        :param int index: Sample index in [0, len(self)].
        :return Dict[str, torch.Tensor]: Sample dictionary with keys 'X' (max_days, d_x),
            'mask' (max_days,), 'y' (T, 1), 't' (T, 1), and 'aux_targets' (T, d_s).
        """
        return self.samples[index]
