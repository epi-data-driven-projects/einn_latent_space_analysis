from typing import Optional

import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork
from einn.model.transformer_attn import TransformerAttn


class FeatureModule(BaseNeuralNetwork):
    """
    Feature Module mapping noisy observations to latent embeddings.
    """
    def __init__(
            self,
            dim_seq_in: int = 10,
            rnn_out: int = 40,
            dim_out: int = 20,
            n_layers: int = 1,
            bidirectional: bool = True,
            dropout: float = 0.0
    ) -> None:
        """
        Initializes the Encoder-Decoder architecture for the Feature Module.

        :param int dim_seq_in: Dimensionality of input data features (d_x).
        :param int rnn_out: Hidden size of the RNN layers.
        :param int dim_out: Final embedding dimension (d_e).
        :param int n_layers: Number of recurrent layers.
        :param bool bidirectional: Whether the Encoder GRU is bidirectional.
        :param float dropout: Dropout probability.
        """
        super(FeatureModule, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1

        # ENCODER
        self.enc_rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // self.num_directions,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=dropout,
            batch_first=True  # Inputs and outputs are [Batch, Seq, Features]
        )
        self.attn_layer = TransformerAttn(dim_in=self.rnn_out, value_dim=self.rnn_out, key_dim=self.rnn_out)
        self.enc_out_layer = nn.Sequential(
            nn.Linear(in_features=self.rnn_out, out_features=self.rnn_out),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # DECODER
        self.dec_rnn = nn.GRU(
            input_size=1,  # time
            hidden_size=self.rnn_out // self.num_directions,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=dropout,
            batch_first=True
        )
        self.dec_out_layer = nn.Sequential(
            nn.Linear(in_features=self.rnn_out, out_features=self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # Initialize linear layers with Xavier uniform
        self.dec_out_layer.apply(self._init_weights)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encodes the input sequence into a fixed-size context vector.
        :param torch.Tensor x: Input sequence. Shape: [Batch, Seq_len, dim_seq_in].
        :param torch.Tensor mask: Optional mask tensor. Shape: [Batch, Seq_len].
        :return torch.Tensor: Context vector (h). Shape: [Batch, rnn_out].
        """
        # latent_seqs shape: [Batch, Seq_len, rnn_out]
        latent_seqs, _ = self.enc_rnn(x)

        if mask is not None:
            latent_seqs = self.attn_layer(latent_seqs, mask)
        else:
            # Apply self-attention -> Shape remains: [Batch, Seq_len, rnn_out]
            latent_seqs = self.attn_layer(latent_seqs)

        # Aggregate over the time dimension (dim=1) to create a single context vector per batch
        # Shape becomes: [Batch, rnn_out]
        latent_seqs = latent_seqs.sum(dim=1)

        # Final projection
        h = self.enc_out_layer(latent_seqs)
        return h

    def decode(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Decodes the context vector across specified time steps.
        :param torch.Tensor h: Context vector from encoder. Shape: [Batch, rnn_out].
        :param torch.Tensor t: Scaled time steps. Shape: [Batch, Seq_len, 1].
        :return: Feature embeddings (e_t^F). Shape: [Batch, Seq_len, dim_out].
        """
        batch_size = h.size(0)
        hidden_size = self.rnn_out // self.num_directions

        # Reshape context vector to match directions: [Batch, num_directions, hidden_size]
        h_reshaped = h.view(batch_size, self.num_directions, hidden_size)

        # Transpose to match GRU's expected h0 shape: [num_layers * num_directions, Batch, hidden_size]
        h0 = h_reshaped.transpose(0, 1).repeat(self.n_layers, 1, 1).contiguous()

        # Pass through Decoder GRU. Output shape: [Batch, Seq_len, rnn_out]
        latent_seqs, _ = self.dec_rnn(t=t, h0=h0)

        # Final projection to embedding space. Shape: [Batch, Seq_len, dim_out]
        e_t_f = self.dec_out_layer(latent_seqs)
        return e_t_f

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Full forward pass mapping sequences to embeddings at given time steps.
        :param torch.Tensor x: Input observations. Shape: [Batch, Seq_len, dim_seq_in].
        :param torch.Tensor t: Target time steps. Shape: [Batch, Seq_len, 1].
        :param torch.Tensor mask: Optional mask tensor. Shape: [Batch, Seq_len].
        :return torch.Tensor: Feature embeddings. Shape: [Batch, Seq_len, dim_out].
        """

        h = self.encode(x=x, mask=mask)
        e_t_f = self.decode(h=h, t=t)

        return e_t_f
