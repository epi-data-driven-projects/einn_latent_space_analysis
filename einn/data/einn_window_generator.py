from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


ArrayLike = Union[np.ndarray, Tensor]


@dataclass
class WindowedEINNData:
    """Windowed arrays produced by :class:`EINNWindowGenerator`.

    All time-series tensors use [n_windows, time, features] layout.
    Masks use 1 for valid values and 0 for padding.
    """

    inputs: Tensor
    input_mask: Tensor
    targets: Optional[Tensor]
    target_mask: Tensor
    target_weights: Optional[Tensor]
    aux_data: Optional[Tensor]
    target_times: Tensor
    end_indices: list[int]

    def as_original_tuple(self):
        """Return the six arrays in the order of the original EINN function."""
        return (
            self.inputs,
            self.input_mask,
            self.targets,
            self.target_mask,
            self.target_weights,
            self.aux_data,
        )


class EINNWindowGenerator:
    """Readable implementation of the original EINN expanding-prefix windowing.

    The input is supplied in [features, time] layout. For every endpoint
    ``end_idx`` in ``[min_sequence_length, ..., sequence_length]``:

    * input: ``X[:end_idx]``; shorter prefixes are right-padded with zero;
    * target / auxiliary window: ``[end_idx-min_sequence_length,
      end_idx+forecast_horizon)``; missing observations at the end are padded
      with ``pad_value``;
    * target mask: one for observations that exist in the original series,
      zero for padded target/auxiliary positions.

    The target-time grid is not padded: it is extrapolated into the requested
    forecast horizon, which allows the Time Module to evaluate future times.
    """

    def __init__(
        self,
        min_sequence_length: int,
        forecast_horizon: int = 56,
        stride: int = 1,
        pad_value: float = -9.0,
        input_pad_value: float = 0.0,
        time_dim: int = -1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min_sequence_length < 1:
            raise ValueError("min_sequence_length must be at least 1.")
        if forecast_horizon < 0:
            raise ValueError("forecast_horizon cannot be negative.")
        if stride < 1:
            raise ValueError("stride must be at least 1.")

        self.min_sequence_length = min_sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.pad_value = pad_value
        self.input_pad_value = input_pad_value
        self.time_dim = time_dim
        self.dtype = dtype

    @property
    def target_length(self) -> int:
        return self.min_sequence_length + self.forecast_horizon

    def _to_time_major(self, data: ArrayLike, name: str) -> Tensor:
        tensor = torch.as_tensor(data, dtype=self.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        elif tensor.ndim == 2:
            tensor = tensor.movedim(self.time_dim % 2, 0)
        else:
            raise ValueError(
                f"{name} must be one- or two-dimensional; got {tuple(tensor.shape)}."
            )
        return tensor.contiguous()

    def _make_prefix_windows(self, data: Tensor, end_indices: list[int]):
        sequences = [data[:end_idx] for end_idx in end_indices]
        masks = [
            torch.ones(end_idx, dtype=self.dtype)
            for end_idx in end_indices
        ]
        return (
            pad_sequence(
                sequences,
                batch_first=True,
                padding_value=self.input_pad_value,
            ),
            pad_sequence(masks, batch_first=True, padding_value=0.0),
        )

    def _make_aligned_windows(
        self,
        data: Optional[Tensor],
        end_indices: list[int],
        sequence_length: int,
    ) -> Optional[Tensor]:
        if data is None:
            return None

        windows = []
        for end_idx in end_indices:
            start_idx = end_idx - self.min_sequence_length
            stop_idx = end_idx + self.forecast_horizon
            observed = data[start_idx:min(stop_idx, sequence_length)]

            padded = torch.full(
                (self.target_length, data.shape[-1]),
                self.pad_value,
                dtype=self.dtype,
            )
            padded[: observed.shape[0]] = observed
            windows.append(padded)

        return torch.stack(windows, dim=0)

    def transform(
        self,
        data: ArrayLike,
        aux_data: Optional[ArrayLike] = None,
        target_data: Optional[ArrayLike] = None,
        target_weights: Optional[ArrayLike] = None,
        time_data: Optional[ArrayLike] = None,
    ) -> WindowedEINNData:
        """Create the expanding-prefix windows for one trajectory.

        Args:
            data: Input features, [features, time].
            aux_data: Auxiliary compartment trajectories, [features, time].
            target_data: Observed target, [target_features, time].
            target_weights: Optional target weights, [target_features, time].
            time_data: Optional physical time values. Defaults to 0, ..., T-1.
        """
        inputs_time_major = self._to_time_major(data, "data")
        sequence_length = inputs_time_major.shape[0]
        if sequence_length < self.min_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} is shorter than minimum "
                f"length {self.min_sequence_length}."
            )

        aligned: dict[str, Optional[Tensor]] = {}
        for name, values in (
            ("aux_data", aux_data),
            ("target_data", target_data),
            ("target_weights", target_weights),
        ):
            tensor = None if values is None else self._to_time_major(values, name)
            if tensor is not None and tensor.shape[0] != sequence_length:
                raise ValueError(
                    f"{name} has {tensor.shape[0]} time steps; expected {sequence_length}."
                )
            aligned[name] = tensor

        if time_data is None:
            time_values = torch.arange(sequence_length, dtype=self.dtype).unsqueeze(-1)
        else:
            time_values = self._to_time_major(time_data, "time_data")
            if time_values.shape[0] != sequence_length or time_values.shape[1] != 1:
                raise ValueError("time_data must contain one time value per observation.")

        end_indices = list(
            range(
                self.min_sequence_length,
                sequence_length + 1,
                self.stride,
            )
        )

        input_windows, input_masks = self._make_prefix_windows(
            inputs_time_major, end_indices
        )

        target_masks = []
        time_windows = []
        for end_idx in end_indices:
            start_idx = end_idx - self.min_sequence_length
            observed_length = min(self.target_length, sequence_length - start_idx)
            mask = torch.zeros(self.target_length, dtype=self.dtype)
            mask[:observed_length] = 1.0
            target_masks.append(mask)

            # The time grid extends beyond the data boundary for forecasting.
            start_time = time_values[start_idx, 0]
            if sequence_length > 1:
                step = time_values[1, 0] - time_values[0, 0]
            else:
                step = torch.tensor(1.0, dtype=self.dtype)
            offsets = torch.arange(self.target_length, dtype=self.dtype) * step
            time_windows.append((start_time + offsets).unsqueeze(-1))

        return WindowedEINNData(
            inputs=input_windows,
            input_mask=input_masks,
            targets=self._make_aligned_windows(
                aligned["target_data"], end_indices, sequence_length
            ),
            target_mask=torch.stack(target_masks, dim=0),
            target_weights=self._make_aligned_windows(
                aligned["target_weights"], end_indices, sequence_length
            ),
            aux_data=self._make_aligned_windows(
                aligned["aux_data"], end_indices, sequence_length
            ),
            target_times=torch.stack(time_windows, dim=0),
            end_indices=end_indices,
        )

    def __call__(self, *args, **kwargs) -> WindowedEINNData:
        return self.transform(*args, **kwargs)
