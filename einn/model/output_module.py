import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork


class OutputModule(BaseNeuralNetwork):
    """
    Shared Multi-Layer Perceptron that decodes latent embeddings (e_t or e_t^F)
    into physical ODE compartment states (e.g., S, E, I, R, M).

    This implementation directly mirrors the deep 4-layer architecture of the
    original EINN codebase.
    """

    def __init__(self, d_e: int = 20, d_s: int = 5) -> None:
        """
        Initializes the Output Module.

        Args:
            d_e (int): Dimension of the input embedding.
            d_s (int): Number of ODE compartment states (out_dim).
        """
        super(OutputModule, self).__init__()

        # Deep MLP architecture mapping embeddings to state space
        # d_e -> 2*d_e -> 2*d_e -> d_e -> d_s
        self.net = nn.Sequential(
            nn.Linear(in_features=d_e, out_features=2 * d_e),
            nn.Tanh(),
            nn.Linear(in_features=2 * d_e, out_features=2 * d_e),
            nn.Tanh(),
            nn.Linear(in_features=2 * d_e, out_features=d_e),
            nn.Tanh(),
            nn.Linear(in_features=d_e, out_features=d_s)
        )

        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Applies Xavier uniform initialization and fills biases with 0.01."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Forward pass decoding the embedding into physical compartment states.
        PyTorch's nn.Linear inherently broadcasts over the Batch and Seq_len dimensions.

        Args:
            e (torch.Tensor): Latent embedding. Shape: [Batch, Seq_len, d_e].

        Returns:
            torch.Tensor: Predicted compartment states. Shape: [Batch, Seq_len, d_s].
        """
        out = self.net(e)
        return out
