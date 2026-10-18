import numpy as np
import torch


class DataPreprocessor:
    """
    Az adatok Z-score skálázásáért és az idő normalizálásáért felelős osztály.
    """

    def __init__(self):
        # Statisztikai paraméterek a skálázáshoz
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.aux_mean = None
        self.aux_std = None
        self.t_max = None

    def fit_transform(self, x_raw, y_raw, aux_raw, target_idx=None) -> dict[str, torch.Tensor]:
        """
        Kiszámolja a statisztikákat (átlag, szórás, max idő), elmenti őket,
        majd leskálázza az adatokat. Csak a legelső (referencia) adathalmazon szabad meghívni!
        """
        # Konvertálás PyTorch tenzorrá, ha NumPy array-t kap
        x = torch.tensor(x_raw, dtype=torch.float32) if isinstance(x_raw,
                                                                   np.ndarray) else x_raw.clone().detach().float()
        y = torch.tensor(y_raw, dtype=torch.float32) if isinstance(y_raw,
                                                                   np.ndarray) else y_raw.clone().detach().float()
        aux = torch.tensor(aux_raw, dtype=torch.float32) if isinstance(aux_raw,
                                                                       np.ndarray) else aux_raw.clone().detach().float()

        # Statisztikák mentése
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0)

        self.y_mean = y.mean(dim=0)
        self.y_std = y.std(dim=0)

        self.aux_mean = aux.mean(dim=0)
        self.aux_std = aux.std(dim=0)

        self.t_max = torch.tensor(float(x.shape[0]), dtype=torch.float32)

        # A skálázást a transform végzi el a kimentett statisztikák alapján
        return self.transform(x_raw, y_raw, aux_raw)

    def transform(self, x_raw, y_raw, aux_raw) -> dict[str, torch.Tensor]:
        """
        Leskálázza az adatokat a KORÁBBAN MENTETT statisztikák alapján.
        Ez garantálja, hogy a teszt/validációs adatok ugyanabba a
        koordinátarendszerbe kerülnek, mint a betanító adatok.
        """
        if self.x_mean is None:
            raise ValueError("A Preprocessor még nem lett illesztve! Futtasd először a fit_transform-ot.")

        x = torch.tensor(x_raw, dtype=torch.float32) if isinstance(x_raw,
                                                                   np.ndarray) else x_raw.clone().detach().float()
        y = torch.tensor(y_raw, dtype=torch.float32) if isinstance(y_raw,
                                                                   np.ndarray) else y_raw.clone().detach().float()
        aux = torch.tensor(aux_raw, dtype=torch.float32) if isinstance(aux_raw,
                                                                       np.ndarray) else aux_raw.clone().detach().float()

        # Z-score normalizáció (1e-8 hozzáadása a 0-val való osztás elkerülésére)
        x_scaled = (x - self.x_mean) / (self.x_std + 1e-8)
        y_scaled = (y - self.y_mean) / (self.y_std + 1e-8)
        aux_scaled = (aux - self.aux_mean) / (self.aux_std + 1e-8)

        # Idő (t) dimenzió létrehozása és [0, 1] tartományba skálázása
        seq_len = x.shape[0]
        t_raw = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        t_scaled = t_raw / self.t_max

        return {
            "x": x_scaled,
            "y": y_scaled,
            "t": t_scaled,
            "aux_targets": aux_scaled
        }
