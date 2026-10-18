import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from typing import Dict


class DataPreprocessor:
    """
    Handles Z-score normalization for all EINN inputs and targets.
    Crucially, it stores the scaling statistics so the EINN can inverse-transform
    the latent states back to the biological domain BEFORE feeding them to the ODE.
    """

    def __init__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.aux_scaler = StandardScaler()

        self.aux_mean: torch.Tensor | None = None
        self.aux_std: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None

        self.t_max = None

    def fit_transform(self, x_raw: np.ndarray, y_raw: np.ndarray, aux_raw: np.ndarray, target_idx: int) -> Dict[
        str, torch.Tensor]:
        """
        Scales the raw data matrices and replaces the target compartment in aux_targets with real incidence data.
        """
        # 1. TRÜKK: Kumulatív rekesz felülírása a napi incidenciával (S_rmse[:, target_idx] = y_raw)
        aux_raw_modified = aux_raw.copy()
        aux_raw_modified[:, target_idx] = y_raw.ravel()

        # 2. Z-score skálázás mindenre
        x_scaled = self.x_scaler.fit_transform(x_raw)
        y_scaled = self.y_scaler.fit_transform(y_raw)
        aux_scaled = self.aux_scaler.fit_transform(aux_raw_modified)

        # 3. Skálázási statisztikák (mean, std) kimentése tenzorként az inverz transzformációhoz
        self.aux_mean = torch.tensor(self.aux_scaler.mean_, dtype=torch.float32)
        self.aux_std = torch.tensor(self.aux_scaler.scale_ + 1e-8, dtype=torch.float32)  # +1e-8 a nullával osztás ellen

        self.y_mean = torch.tensor(self.y_scaler.mean_, dtype=torch.float32)
        self.y_std = torch.tensor(self.y_scaler.scale_ + 1e-8, dtype=torch.float32)

        t_unscaled = np.arange(len(x_raw)).reshape(-1, 1)

        # Mentsük el a maximumot egy tenzorba a Forward Engine számára
        self.t_max = torch.tensor(t_unscaled.max(), dtype=torch.float32)

        return {
            "x": torch.tensor(x_scaled, dtype=torch.float32),
            "y": torch.tensor(y_scaled, dtype=torch.float32),
            "t": torch.tensor(t_unscaled, dtype=torch.float32),  # Nyers idő megy a Datasetbe!
            "aux_targets": torch.tensor(aux_scaled, dtype=torch.float32)
        }