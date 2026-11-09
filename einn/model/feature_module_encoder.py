from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        """Encode a padded variable-length input and return one context vector.

        The window generator uses prefix padding: valid observations come first,
        followed by zero padding. The GRU is packed so the reverse direction does
        not read padded positions. Padded query outputs are also excluded from the
        attention pooling operation.
        """
        if mask is not None:
            lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
            packed_x = pack_padded_sequence(
                x,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.enc_rnn(packed_x)
            latent_seqs, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.shape[1],
            )

            key_padding_mask = (mask == 0).bool()
            latent_seqs, _ = self.attn_layer(
                query=latent_seqs,
                key=latent_seqs,
                value=latent_seqs,
                key_padding_mask=key_padding_mask,
            )
            latent_seqs = latent_seqs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        else:
            latent_seqs, _ = self.enc_rnn(input=x)
            latent_seqs, _ = self.attn_layer(
                query=latent_seqs,
                key=latent_seqs,
                value=latent_seqs,
            )

        latent_seqs = latent_seqs.sum(dim=1)
        return self.enc_out_layer(input=latent_seqs)
