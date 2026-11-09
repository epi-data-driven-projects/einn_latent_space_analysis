from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from einn.data.einn_window_generator import EINNWindowGenerator


class WindowedEINNDataset(Dataset):
    """Dataset adapter that feeds expanding-prefix windows to the current EINNTrainer.

    Each item contains:
        X:             [max_input_length, d_x]
        mask:          [max_input_length], 1 = valid input, 0 = padding
        y:             [min_sequence_length + horizon, 1]
        t:             [min_sequence_length + horizon, 1]
        aux_targets:   [min_sequence_length + horizon, d_s]
        target_mask:   [min_sequence_length + horizon], 1 = observed, 0 = padding
        history_steps: integer separating the historical overlap from the future
    """

    def __init__(
        self,
        x_series: Sequence[np.ndarray],
        y_series: Sequence[np.ndarray],
        aux_series: Sequence[np.ndarray],
        window_generator: EINNWindowGenerator,
    ) -> None:
        if not (len(x_series) == len(y_series) == len(aux_series)):
            raise ValueError("x_series, y_series and aux_series must have equal lengths.")
        if not x_series:
            raise ValueError("At least one trajectory is required.")

        self.samples: list[dict[str, torch.Tensor]] = []
        self.history_steps = window_generator.min_sequence_length

        for series_idx, (x, y, aux) in enumerate(zip(x_series, y_series, aux_series)):
            if x.shape[0] != y.shape[0] or x.shape[0] != aux.shape[0]:
                raise ValueError(
                    f"Trajectory {series_idx} has inconsistent time dimensions: "
                    f"X={x.shape}, y={y.shape}, aux={aux.shape}."
                )

            windows = window_generator(
                data=x.T,
                target_data=y.T,
                aux_data=aux.T,
            )

            for window_idx in range(len(windows.end_indices)):
                self.samples.append(
                    {
                        "X": windows.inputs[window_idx].float(),
                        "mask": windows.input_mask[window_idx].float(),
                        "y": windows.targets[window_idx].float(),
                        "t": windows.target_times[window_idx].float(),
                        "aux_targets": windows.aux_data[window_idx].float(),
                        "target_mask": windows.target_mask[window_idx].float(),
                        "history_steps": torch.tensor(self.history_steps, dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]
