from typing import Optional

import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork


class FeatureModuleEncoder(BaseNeuralNetwork):
    """
    Encoder part of the Feature Module mapping noisy observations to a context vector.
    """
    def __init__(
            self,
            dim_seq_in: int = 10,
            rnn_out: int = 40,
            n_layers: int = 1,
            bidirectional: bool = True,
            dropout: float = 0.0
    ):
        """
        Initializes the Encoder architecture.

        :param int dim_seq_in: Dimensionality of input data features (d_x).
        :param int rnn_out: Hidden size of the RNN layers.
        :param int n_layers: Number of recurrent layers.
        :param bool bidirectional: Whether the Encoder GRU is bidirectional.
        :param float dropout: Dropout probability.
        """
        super().__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1

        self.enc_rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // self.num_directions,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=dropout,
            batch_first=True  # Inputs and outputs are [Batch, Seq, Features]
        )
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=self.rnn_out,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )
        self.enc_out_layer = nn.Sequential(
            nn.Linear(in_features=self.rnn_out, out_features=self.rnn_out),
            nn.Tanh(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encodes the input sequence into a fixed-size context vector.
        :param torch.Tensor x: Input sequence. Shape: [Batch, Seq_len, dim_seq_in].
        :param torch.Tensor mask: Optional mask tensor. Shape: [Batch, Seq_len].
        :return torch.Tensor: Context vector (h). Shape: [Batch, rnn_out].
        """
        # latent_seqs shape: [Batch, Seq_len, rnn_out]
        latent_seqs, _ = self.enc_rnn(input=x)

        if mask is not None:
            # PyTorch's MultiheadAttention expects True for elements that should be ignored.
            pytorch_mask = (mask == 0).bool()

            latent_seqs, _ = self.attn_layer(
                query=latent_seqs,
                key=latent_seqs,
                value=latent_seqs,
                key_padding_mask=pytorch_mask
            )
        else:
            latent_seqs, _ = self.attn_layer(
                query=latent_seqs,
                key=latent_seqs,
                value=latent_seqs
            )

        # Aggregate over the time dimension (dim=1) to create a single context vector per batch
        # Shape becomes: [Batch, rnn_out]
        latent_seqs = latent_seqs.sum(dim=1)

        # Final projection
        h = self.enc_out_layer(input=latent_seqs)
        return h
