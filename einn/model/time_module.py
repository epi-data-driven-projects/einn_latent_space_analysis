import numpy as np
import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork


class TimeModule(BaseNeuralNetwork):
    """Original EINN Fourier time encoder, without a projection head."""

    def __init__(
        self,
        mapping_size: int = 20,
        scale: float = 1.0,
        out_dim: int = 20,
        seed: int = 42,
    ):
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
        x_proj = (2.0 * np.pi * t) @ self.b_gauss.T
        return torch.cat((torch.sin(x_proj), torch.cos(x_proj)), dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.apply_fourier_mapping(t))
