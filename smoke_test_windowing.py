"""Standalone shape check for the new windowing code."""

import numpy as np
from torch.utils.data import DataLoader

from einn.data.einn_window_generator import EINNWindowGenerator
from einn.data.windowed_einn_dataset import WindowedEINNDataset


rng = np.random.default_rng(42)
x = rng.normal(size=(200, 5)).astype(np.float32)
y = rng.normal(size=(200, 1)).astype(np.float32)
aux = rng.normal(size=(200, 5)).astype(np.float32)

generator = EINNWindowGenerator(
    min_sequence_length=10,
    forecast_horizon=56,
    stride=1,
)
windows = generator(data=x.T, target_data=y.T, aux_data=aux.T)
assert windows.inputs.shape == (191, 200, 5)
assert windows.targets.shape == (191, 66, 1)
assert windows.aux_data.shape == (191, 66, 5)
assert windows.target_times.shape == (191, 66, 1)
assert windows.target_mask[-1].sum().item() == 10

dataset = WindowedEINNDataset([x], [y], [aux], generator)
batch = next(iter(DataLoader(dataset, batch_size=4)))
assert batch["X"].shape == (4, 200, 5)
assert batch["y"].shape == (4, 66, 1)
assert batch["t"].shape == (4, 66, 1)
print("Windowing smoke test passed.")