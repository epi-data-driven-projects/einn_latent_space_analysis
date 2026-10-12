import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from typing import Dict


class DataPreprocessor:
    def __init__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.aux_scaler = StandardScaler()

        self.aux_mean: torch.Tensor | None = None
        self.aux_std: torch.Tensor | None = None

    def fit_transform(self, x_raw: np.ndarray, y_raw: np.ndarray, aux_raw: np.ndarray, target_idx: int) -> Dict[
        str, torch.Tensor]:
        # 1. A TRÜKK: Az offline ODE kalibráció (aux_raw) célváltozó oszlopát
        # felülírjuk a hajszálpontos valós adatokkal (y_raw), ahogy az eredeti cikk tette!
        aux_raw_modified = aux_raw.copy()
        aux_raw_modified[:, target_idx] = y_raw.ravel()

        # 2. Z-score skálázás mindenre
        x_scaled = self.x_scaler.fit_transform(x_raw)
        y_scaled = self.y_scaler.fit_transform(y_raw)
        aux_scaled = self.aux_scaler.fit_transform(aux_raw_modified)

        # 3. Kimentjük az aux (fizikai rekeszek) szórását és átlagát a Loss függvény számára!
        # Ez kritikus a fizikai egyenletekhez való visszatranszformáláshoz.
        self.aux_mean = torch.tensor(self.aux_scaler.mean_, dtype=torch.float32)
        # Hozzáadunk egy pici epszilont, hogy sose osszunk nullával
        self.aux_std = torch.tensor(self.aux_scaler.scale_ + 1e-7, dtype=torch.float32)

        # 4. Idővektor (t) [0, 1] tartományra
        t_raw = np.arange(len(x_raw)).reshape(-1, 1)
        t_scaled = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min() + 1e-8)

        return {
            "x": torch.tensor(x_scaled, dtype=torch.float32),
            "y": torch.tensor(y_scaled, dtype=torch.float32),
            "t": torch.tensor(t_scaled, dtype=torch.float32),
            "aux_targets": torch.tensor(aux_scaled, dtype=torch.float32)
        }
