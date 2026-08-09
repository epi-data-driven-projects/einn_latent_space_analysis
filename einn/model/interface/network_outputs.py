from dataclasses import dataclass
import torch


@dataclass
class NetworkOutputs:
    """
    A neurális hálózatok kimeneteit összefogó adatcsomag.
    """
    e_t: torch.Tensor  # Time modul beágyazása
    e_t_F: torch.Tensor  # Feature modul beágyazása
    s_t: torch.Tensor  # Time modul alapján becsült állapotok
    s_t_F: torch.Tensor  # Feature modul alapján becsült állapotok
