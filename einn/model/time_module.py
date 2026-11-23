import numpy as np
import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork


class TimeModule(BaseNeuralNetwork):
    """
    Time Module that applies a Fourier feature mapping to the time vector
    and processes it through a Multi-Layer Perceptron.
    """

    def __init__(self, mapping_size: int = 20, scale: float = 1.0, out_dim: int = 20, seed: int = 42):
        """
        Initializes the Time Module with a fixed Gaussian matrix for Fourier mapping.
        :param int mapping_size: Size of the Fourier mapping.
        :param float scale: Scale parameter for the Gaussian distribution.
        :param int out_dim: Final output dimension (d_e).
        :param int seed: Random seed for reproducible b_gauss generation.
        """
        super().__init__()
        rng = np.random.default_rng(seed)
        b_gauss = rng.normal(size=(mapping_size, 1)) * scale
        self.register_buffer("b_gauss", torch.from_numpy(b_gauss).float())
        self.out_dim = int(out_dim)

        hidden_dim = 2 * mapping_size
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

        self.net.apply(self._init_weights)

    def apply_fourier_mapping(self, t: torch.Tensor) -> torch.Tensor:
        """
        Applies Fourier feature mapping to the scaled time vector.
        :param torch.Tensor t: Time vector. Shape: [Batch, Seq_len, 1].
        :return torch.Tensor: Mapped features. Shape: [Batch, Seq_len, 2 * mapping_size].
        """

        if self.b_gauss is None:
            return t


        :param torch.Tensor t: Time coordinates of the modeled trajectory.
        :return torch.Tensor: Tensor containing the computed model outputs or time derivatives.
        """
        x_proj = (2.0 * np.pi * t) @ self.b_gauss.T
        return torch.cat((torch.sin(x_proj), torch.cos(x_proj)), dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        return self.net(self.apply_fourier_mapping(t))
