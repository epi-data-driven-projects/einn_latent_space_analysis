"""
Expanding-window PyTorch Dataset with true and assumed model labels for multi-model EINN.
"""
from typing import Dict, List, Optional

import numpy as np
import torch

from einn.data.einn_dataset import EINNDataset


class ExpandingWindowLoader(EINNDataset):
    """
    PyTorch Dataset of zero-padded expanding prefixes with epidemiological model labels.

    Extends the single-model expanding-window scheme (thesis Section 2.3) to multi-model
    training (Section 2.4) and mismatched-model control experiments (Section 5). Follows the
    project mask convention where 0.0 marks observed timesteps and 1.0 marks padded future
    timesteps, while full-horizon targets (y, t, aux_targets) remain unmasked.
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
        device: str = "cuda:1",
        true_model_labels: Optional[List[int]] = None,
        assumed_model_labels: Optional[List[int]] = None,
    ) -> None:
        """
        Validate base trajectories and pre-compute all expanding-prefix samples with model labels.

        Also retains the unwindowed preprocessed lists on the instance for held-out inference,
        PCA projection, and physical-scale forecast evaluation.

        :param List[np.ndarray] x_list: Standardized input matrices, each of shape (T, d_x) with T >= max_days.
        :param List[np.ndarray] y_list: Standardized target series, each of shape (T, 1).
        :param List[np.ndarray] t_list: Unscaled time index grids, each of shape (T, 1).
        :param List[np.ndarray] aux_list: Standardized auxiliary compartment trajectories, each of shape (T, d_s).
        :param int min_days: Shortest observation prefix length included in training (>= 1).
        :param int max_days: Padded sequence length and longest observation prefix (>= min_days).
        :param int step: Increment between consecutive expanding-window prefix lengths (>= 1).
        :param str device: Target PyTorch device string (e.g. cpu or cuda:0).
        :param Optional[List[int]] true_model_labels: Ground-truth generating model IDs a_i (one per trajectory);
            defaults to all zeros if None.
        :param Optional[List[int]] assumed_model_labels: Hypothesized model IDs passed to the network
            (one per trajectory); defaults to a copy of true_model_labels if None.
        :return None: Populates self.samples and stores unwindowed trajectory metadata in place.
        :raises ValueError: If list lengths mismatch, window parameters are invalid, or any
            trajectory has fewer than max_days timesteps.
        """
        lengths = {len(x_list), len(y_list), len(t_list), len(aux_list)}
        if len(lengths) != 1:
            raise ValueError("x_list, y_list, t_list and aux_list must have equal lengths.")
        n = len(x_list)
        if true_model_labels is None:
            true_model_labels = [0] * n
        if assumed_model_labels is None:
            assumed_model_labels = list(true_model_labels)
        if len(true_model_labels) != n or len(assumed_model_labels) != n:
            raise ValueError("Model-label lists must have one item per base trajectory.")
        if min_days < 1 or max_days < min_days or step < 1:
            raise ValueError("Require 1 <= min_days <= max_days and step >= 1.")

        # Retain unwindowed trajectories and labels for downstream evaluation and latent-space analysis.
        self.x_list = list(x_list)
        self.y_list = list(y_list)
        self.t_list = list(t_list)
        self.aux_list = list(aux_list)
        self.true_model_labels = [int(x) for x in true_model_labels]
        self.assumed_model_labels = [int(x) for x in assumed_model_labels]

        self.device = torch.device(device)
        self.samples: list[dict[str, torch.Tensor]] = []

        # Build nested zero-padded prefixes for each trajectory and attach scalar model-label tensors.
        for x, y, t, aux, true_label, assumed_label in zip(
            x_list, y_list, t_list, aux_list,
            true_model_labels, assumed_model_labels,
        ):
            if len(x) < max_days or len(y) < max_days or len(t) < max_days or len(aux) < max_days:
                raise ValueError("Every base trajectory must contain at least max_days rows.")
            for current_days in range(min_days, max_days + 1, step):
                # Copy observed prefix [0, current_days) and zero-pad unobserved steps to (max_days, d_x).
                x_padded = torch.zeros(
                    (max_days, x.shape[1]), dtype=torch.float32, device=self.device
                )
                x_padded[:current_days] = torch.as_tensor(
                    x[:current_days], dtype=torch.float32, device=self.device
                )

                # Padding mask of shape (max_days,): 0.0 = observed, 1.0 = padded future.
                mask = torch.ones(max_days, dtype=torch.float32, device=self.device)
                mask[:current_days] = 0.0

                # Scalar int64 model IDs and boolean mismatch flag for multi-model routing and loss masking.
                true_tensor = torch.tensor(int(true_label), dtype=torch.long, device=self.device)
                assumed_tensor = torch.tensor(int(assumed_label), dtype=torch.long, device=self.device)

                self.samples.append({
                    "X": x_padded,                                                         # Shape: (max_days, d_x)
                    "mask": mask,                                                          # Shape: (max_days,)
                    "y": torch.as_tensor(y, dtype=torch.float32, device=self.device),      # Shape: (T, 1)
                    "t": torch.as_tensor(t, dtype=torch.float32, device=self.device),      # Shape: (T, 1)
                    "aux_targets": torch.as_tensor(aux, dtype=torch.float32, device=self.device),  # Shape: (T, d_s)
                    "true_model_label": true_tensor,                                       # Shape: ()
                    "assumed_model_label": assumed_tensor,                                 # Shape: ()
                    "is_mismatch": true_tensor != assumed_tensor,                          # Shape: ()
                })

    def __len__(self) -> int:
        """
        Return the total number of expanding-window samples across all trajectories.

        :return int: Total number of pre-computed (trajectory, prefix_length) samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single pre-computed expanding-window sample by index.

        :param int index: Zero-based sample index in [0, len(self)).
        :return Dict[str, torch.Tensor]: Sample dictionary with keys X (max_days, d_x),
            mask (max_days,), y (T, 1), t (T, 1), aux_targets (T, d_s),
            true_model_label (), assumed_model_label (), and is_mismatch ().
        """
        return self.samples[index]
