import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork


class OutputModule(BaseNeuralNetwork):
    """
    Shared Multi-Layer Perceptron that decodes latent embeddings (e_t or e_t^F)
    into physical ODE compartment states (e.g., S, E, I, R, M).
    """

    def __init__(self, d_e: int = 20, d_s: int = 5) -> None:
        """
        Initializes the Output Module.

        :param int d_e: Dimension of the input embedding.
        :param int d_s: Number of ODE compartment states.
        """
        super().__init__()

        # Deep MLP architecture mapping embeddings to state space
        # d_e -> 2 * d_e -> 2 * d_e -> d_e -> d_s
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

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Forward pass decoding the embedding into physical compartment states.
        PyTorch's nn.Linear inherently broadcasts over the Batch and Seq_len dimensions.
        :param torch.Tensor e: Latent embedding. Shape: [Batch, Seq_len, d_e].
        :return torch.Tensor: Predicted compartment states. Shape: [Batch, Seq_len, d_s].
        """

        return self.net(e)
