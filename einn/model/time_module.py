import numpy as np
import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork


class TimeModule(BaseNeuralNetwork):
    """
    Time Module that applies a Fourier feature mapping to the time vector
    and processes it through a Multi-Layer Perceptron.
    Equivalent to the original 'ff_net_fourier'.
    """

    def __init__(self, mapping_size: int = 20, scale: float = 1.0, out_dim: int = 20, seed: int = 42) -> None:
        """
        Initializes the Time Module with a fixed Gaussian matrix for Fourier mapping.

        Args:
            mapping_size (int): Size of the Fourier mapping.
            scale (float): Scale parameter for the Gaussian distribution.
            out_dim (int): Final output dimension (d_e).
            seed (int): Random seed for reproducible B_gauss generation.
        """
        super(TimeModule, self).__init__()

        # Generate and register the fixed Gaussian matrix B_gauss.
        # register_buffer ensures B_gauss automatically moves to the GPU when .to(device) is called.
        np.random.seed(seed)
        B_gauss = np.random.normal(size=(mapping_size, 1)) * scale
        self.register_buffer('B_gauss', torch.from_numpy(B_gauss).float())

        hidden_dim = 2 * mapping_size
        act_fcn = nn.Tanh()

        # 4-layer MLP as per the original implementation
        self.net = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fcn,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fcn,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fcn,
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            act_fcn
        )

        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Applies Xavier uniform initialization and fills biases with 0.01."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def apply_fourier_mapping(self, t: torch.Tensor) -> torch.Tensor:
        """
        Applies Fourier feature mapping to the scaled time vector.

        Args:
            t (torch.Tensor): Time steps. Shape: [Batch, Seq_len, 1].

        Returns:
            torch.Tensor: Mapped features. Shape: [Batch, Seq_len, 2 * mapping_size].
        """
        if self.B_gauss is None:
            return t

        # t is [Batch, Seq_len, 1], B_gauss.T is [1, mapping_size].
        # Resulting x_proj is [Batch, Seq_len, mapping_size].
        x_proj = (2. * np.pi * t) @ self.B_gauss.T

        # Concatenate sin and cos along the last dimension.
        # Final shape: [Batch, Seq_len, 2 * mapping_size].
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Time Module.

        Args:
            t (torch.Tensor): Time steps. Shape: [Batch, Seq_len, 1].

        Returns:
            torch.Tensor: Time-based embedding (e_t). Shape: [Batch, Seq_len, out_dim].
        """
        mapped_inputs = self.apply_fourier_mapping(t)
        emb_E = self.net(mapped_inputs)
        return emb_E
