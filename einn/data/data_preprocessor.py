from typing import Tuple, Dict

import torch

from einn.ode.base_ode_model import BaseODEModel


class DataPreprocessor:
    """
    Class for scaling the data and generating auxiliary trajectories
    for a given physical ODE model.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the DataPreprocessor.

        Args:
            device (str): The computing device ('cpu' or 'cuda').
        """
        self.scalers: Dict[str, torch.Tensor] = {}
        self.device = torch.device(device)

    def scale_data(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scales the input features and target variable using Z-score normalization,
        and scales the time vector to the [0, 1] range using Min-Max scaling.

        Args:
            x (torch.Tensor): Observed features. Expected shape: [Batch, Seq_len, d_x].
            y (torch.Tensor): Target variable (e.g., mortality). Expected shape: [Batch, Seq_len, 1].
            t (torch.Tensor): Time vector. Expected shape: [Batch, Seq_len, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Scaled x, y, and t tensors moved to the configured device.
        """
        # Move tensors to the specified device (GPU) for faster tensor operations
        x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)

        # Scale target variable (y) using Z-score normalization
        y_mean, y_std = y.mean(), y.std()
        y_scaled = (y - y_mean) / (y_std + 1e-8)
        self.scalers['y_mean'] = y_mean
        self.scalers['y_std'] = y_std

        # Scale features (x) using Z-score normalization.
        # dim=1 corresponds to the Seq_len, meaning we normalize across the time dimension.
        x_mean, x_std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
        x_scaled = (x - x_mean) / (x_std + 1e-8)
        self.scalers['X_mean'] = x_mean
        self.scalers['X_std'] = x_std

        # Scale time (t) to [0, 1] range using Min-Max scaling
        t_min, t_max = t.min(), t.max()
        t_scaled = (t - t_min) / (t_max - t_min + 1e-8)
        self.scalers['t_min'] = t_min
        self.scalers['t_max'] = t_max

        return x_scaled, y_scaled, t_scaled

    def inverse_scale_predictions(self, s_future: torch.Tensor) -> torch.Tensor:
        """
        Rescales the normalized predictions back to their original physical scale.

        Args:
            s_future (torch.Tensor): Predicted normalized states. Shape: [Batch, Seq_len, d_s].

        Returns:
            torch.Tensor: Predictions in original scale. Shape: [Batch, Seq_len, d_s].
        """
        if 'y_mean' not in self.scalers or 'y_std' not in self.scalers:
            raise ValueError("Method 'scale_data' must be executed before inverse scaling.")

        # Revert the Z-score transformation: X = (X_scaled * std) + mean
        return (s_future * self.scalers['y_std']) + self.scalers['y_mean']

    def generate_aux_targets(self, t: torch.Tensor, ode_model: BaseODEModel, d_s: int = 5) -> torch.Tensor:
        """
        Generates auxiliary targets (ideal trajectories) for the given ODE model.

        Args:
            t (torch.Tensor): Time vector. Shape: [Batch, Seq_len, 1].
            ode_model (BaseODEModel): The physical model.
            d_s (int): Number of ODE compartment states. Defaults to 5 (SEIRM).

        Returns:
            torch.Tensor: Auxiliary targets. Shape: [Batch, Seq_len, d_s].
        """
        print("Attention: Generation of aux targets is currently in mock mode.")
        # TODO: Implement a real ODE solver (e.g., scipy or torchdiffeq) with initial parameters.

        batch_size = t.shape[0]
        seq_len = t.shape[1]

        # Output is created directly on the target device
        return torch.zeros(batch_size, seq_len, d_s, device=self.device)
