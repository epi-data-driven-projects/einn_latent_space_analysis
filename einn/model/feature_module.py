from typing import Optional

import torch

from einn.model.base_neural_network import BaseNeuralNetwork
from einn.model.feature_module_decoder import FeatureModuleDecoder
from einn.model.feature_module_encoder import FeatureModuleEncoder


class FeatureModule(BaseNeuralNetwork):
    """
    Feature Module coordinator mapping noisy observations to latent embeddings
    by chaining the Encoder and Decoder.
    """
    def __init__(
            self,
            dim_seq_in: int = 10,
            rnn_out: int = 40,
            dim_out: int = 20,
            n_layers: int = 1,
            bidirectional: bool = True,
            dropout: float = 0.0
    ):
        """
        Initializes the Encoder-Decoder architecture for the Feature Module.

        :param int dim_seq_in: Dimensionality of input data features (d_x).
        :param int rnn_out: Hidden size of the RNN layers.
        :param int dim_out: Final embedding dimension (d_e).
        :param int n_layers: Number of recurrent layers.
        :param bool bidirectional: Whether the GRUs are bidirectional.
        :param float dropout: Dropout probability.
        """
        super().__init__()

        self.encoder = FeatureModuleEncoder(
            dim_seq_in=dim_seq_in,
            rnn_out=rnn_out,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

        self.decoder = FeatureModuleDecoder(
            rnn_out=rnn_out,
            dim_out=dim_out,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Full forward pass mapping sequences to embeddings at given time steps.
        :param torch.Tensor x: Input observations. Shape: [Batch, Seq_len, dim_seq_in].
        :param torch.Tensor t: Target time steps. Shape: [Batch, Seq_len, 1].
        :param torch.Tensor mask: Optional mask tensor. Shape: [Batch, Seq_len].
        :return torch.Tensor: Feature embeddings. Shape: [Batch, Seq_len, dim_out].
        """
        h = self.encoder(x=x, mask=mask)
        e_t_f = self.decoder(h=h, t=t)

        return e_t_f
