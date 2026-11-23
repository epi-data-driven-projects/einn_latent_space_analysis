from typing import Dict, List, Optional

import numpy as np
import torch

from einn.data.einn_dataset import EINNDataset


class ExpandingWindowLoader(EINNDataset):
    """Dataset of zero-padded expanding prefixes with true/assumed model labels.

    Mask convention is inherited from the existing project: 0 means observed and
    1 means padding. Targets, times and auxiliary trajectories retain full length.
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
        device: str = "cpu",
        true_model_labels: Optional[List[int]] = None,
        assumed_model_labels: Optional[List[int]] = None,
    ):

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

        # Retain the unwindowed, preprocessed trajectories for held-out inference,
        # PCA, Hausdorff and physical-scale forecast evaluation.
        self.x_list = list(x_list)
        self.y_list = list(y_list)
        self.t_list = list(t_list)
        self.aux_list = list(aux_list)
        self.true_model_labels = [int(x) for x in true_model_labels]
        # Legacy compatibility key only: true-label-only mode makes it identical.
        self.assumed_model_labels = [int(x) for x in assumed_model_labels]

        self.device = torch.device(device)
        self.samples: list[dict[str, torch.Tensor]] = []

        for x, y, t, aux, true_label, assumed_label in zip(
            x_list, y_list, t_list, aux_list,
            true_model_labels, assumed_model_labels,
        ):
            if len(x) < max_days or len(y) < max_days or len(t) < max_days or len(aux) < max_days:
                raise ValueError("Every base trajectory must contain at least max_days rows.")
            for current_days in range(min_days, max_days + 1, step):
                x_padded = torch.zeros(
                    (max_days, x.shape[1]), dtype=torch.float32, device=self.device
                )
                x_padded[:current_days] = torch.as_tensor(
                    x[:current_days], dtype=torch.float32, device=self.device
                )
                mask = torch.ones(max_days, dtype=torch.float32, device=self.device)
                mask[:current_days] = 0.0

                true_tensor = torch.tensor(int(true_label), dtype=torch.long, device=self.device)
                assumed_tensor = torch.tensor(int(assumed_label), dtype=torch.long, device=self.device)
                self.samples.append({
                    "X": x_padded,
                    "mask": mask,
                    "y": torch.as_tensor(y, dtype=torch.float32, device=self.device),
                    "t": torch.as_tensor(t, dtype=torch.float32, device=self.device),
                    "aux_targets": torch.as_tensor(aux, dtype=torch.float32, device=self.device),
                    "true_model_label": true_tensor,
                    "assumed_model_label": assumed_tensor,
                    "is_mismatch": true_tensor != assumed_tensor,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]
