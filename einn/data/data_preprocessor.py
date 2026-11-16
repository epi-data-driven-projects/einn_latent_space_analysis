from typing import Dict, Tuple, List

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from einn.data.data_scaling_stats import DataScalingStats


class DataPreprocessor:
    """
    Handles Z-score normalization (Standardization) for all EINN inputs and targets.
    Crucially, it generates and returns the DataScalingStats object, which allows the EINNForwardEngine 
    to inverse-transform the neural latent states back to the biological domain dynamically.
    """

    def __init__(self):
        """
        Initializes the preprocessor with independent scikit-learn StandardScalers 
        for the exogenous features, targets, and auxiliary curves.
        """
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.aux_scaler = StandardScaler()

        # Placeholders for the PyTorch-compatible scaling statistics
        self.aux_mean: torch.Tensor | None = None
        self.aux_std: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None

        self.t_min: torch.Tensor | None = None
        self.t_max: torch.Tensor | None = None

    def fit_transform(
        self, 
        x_raw: np.ndarray, 
        y_raw: np.ndarray, 
        aux_raw: np.ndarray, 
        target_idx: int,
        initial_states: List[float],
        time_length: int
    ) -> Tuple[Dict[str, torch.Tensor], DataScalingStats]:
        """
        Scales the raw data matrices, anchors the pre-calibrated auxiliary structural data,
        and constructs the exact statistical dataclass required by the trainer.
        """
        # Overwrite the targeted compartment in the auxiliary data with the real-world incidence data
        aux_raw_modified = aux_raw.copy()
        aux_raw_modified[:, target_idx] = y_raw.ravel()

        # Apply independent Z-score scaling to all data matrices
        x_scaled = self.x_scaler.fit_transform(x_raw)
        y_scaled = self.y_scaler.fit_transform(y_raw)
        aux_scaled = self.aux_scaler.fit_transform(aux_raw_modified)

        # Extract and store the scaling statistics as PyTorch tensors
        self.aux_mean = torch.tensor(self.aux_scaler.mean_, dtype=torch.float32)
        self.aux_std = torch.tensor(self.aux_scaler.scale_ + 1e-8, dtype=torch.float32)
        self.y_mean = torch.tensor(self.y_scaler.mean_, dtype=torch.float32)
        self.y_std = torch.tensor(self.y_scaler.scale_ + 1e-8, dtype=torch.float32)

        self.t_min = torch.tensor(0.0, dtype=torch.float32)
        self.t_max = torch.tensor(float(time_length - 1), dtype=torch.float32)

        t_unscaled = np.arange(len(x_raw)).reshape(-1, 1)

        # Normalize the initial states using the computed auxiliary statistics
        ic_physical = torch.tensor(initial_states, dtype=torch.float32)
        ic_normalized = (ic_physical - self.aux_mean) / self.aux_std

        # Pack the runtime statistics dataclass
        scaling_stats = DataScalingStats(
            aux_mean=self.aux_mean,
            aux_std=self.aux_std,
            t_min=self.t_min,
            t_max=self.t_max,
            initial_states=ic_normalized.cpu().tolist()
        )

        scaled_data_dict = {
            "x": torch.tensor(x_scaled, dtype=torch.float32),
            "y": torch.tensor(y_scaled, dtype=torch.float32),
            "t": torch.tensor(t_unscaled, dtype=torch.float32),
            "aux_targets": torch.tensor(aux_scaled, dtype=torch.float32)
        }

        return scaled_data_dict, scaling_stats
