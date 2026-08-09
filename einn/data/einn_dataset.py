import torch
from torch.utils.data import Dataset


class EINNDataset(Dataset):
    """
    PyTorch Dataset az EINN számára.
    Alapértelmezetten a teljes idősort egyetlen batch-ként kezeli.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, aux_targets: torch.Tensor):
        self.x = x.float()
        self.y = y.float()
        self.t = t.float()  # esetleg detach? - lehet butaság
        self.aux_targets = aux_targets.float()

    def __len__(self):
        # Mivel a teljes görbét egyben adjuk be, a hossz 1.
        # Csúszóablak esetén ez az ablakok száma lenne.
        return 1  # TODO: ezt is általánosítva, hogy a batch_size-t adja vissza

    def __getitem__(self, idx):
        # TODO: általánosítani!!!
        # Jelenleg az idx-et ignoráljuk, mert egy batch van.
        # Csúszóablak logikánál itt végeznénk a szeletelést (slicing).
        return {
            'X': self.x,
            'y': self.y,
            't': self.t,
            'aux_targets': self.aux_targets
        }