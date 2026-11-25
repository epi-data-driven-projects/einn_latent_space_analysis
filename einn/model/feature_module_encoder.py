from typing import Optional

import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork


class FeatureModuleEncoder(BaseNeuralNetwork):
    """
    Encoder part of the Feature Module mapping noisy observations to a context vector.

    This module is responsible for processing the noisy, multivariate synthetic data
    (the observed epidemic curves and correlated covariates) and compressing the
    historical temporal information into a single, fixed-size context vector (h).
    This context vector is subsequently used by the Decoder to initialize its hidden state.
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

        :param int dim_seq_in: Dimensionality of input data features (d_x). This corresponds
            to the number of input channels (e.g., noisy observations + covariates).
        :param int rnn_out: Hidden size of the RNN layers and the resulting context vector.
        :param int n_layers: Number of recurrent layers in the GRU.
        :param bool bidirectional: Whether the Encoder GRU is bidirectional. If True,
            the hidden size is split equally between forward and backward directions.
        :param float dropout: Dropout probability applied to the GRU, Attention, and output layers.
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
            batch_first=True  # Inputs and outputs are formatted as [Batch, Seq, Features]
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

        The method processes the input sequence through a GRU, applies a self-attention
        mechanism, and aggregates the valid (unmasked) time steps into a single
        summary vector.

        :param torch.Tensor x: Input sequence containing noisy observations and covariates.
            Shape: [Batch, Seq_len, dim_seq_in].
        :param Optional[torch.Tensor] mask: Optional mask tensor indicating the valid
            time window. Expected format: 0 for valid observed days, 1 for masked/future days.
            Shape: [Batch, Seq_len].
        :return torch.Tensor: The aggregated context vector (h). Shape: [Batch, rnn_out].
        """
        # latent_seqs shape: [Batch, Seq_len, rnn_out]
        latent_seqs, _ = self.enc_rnn(input=x)

        key_padding_mask = None

        if mask is not None:
            # Note: PyTorch's MultiheadAttention expects True for masked positions
            key_padding_mask = mask.bool()

        # Apply self-attention over the sequence
        latent_seqs, _ = self.attn_layer(
            query=latent_seqs,
            key=latent_seqs,
            value=latent_seqs,
            key_padding_mask=key_padding_mask
        )

        if mask is not None:
            # Manual masking after the Attention layer
            # Convert mask logic: 1 indicates valid (observed), 0 indicates padding (future/masked)
            # valid_mask shape: [Batch, Seq_len, 1]
            valid_mask = (1 - mask).unsqueeze(-1)

            # Zero out the hidden states corresponding to the padded/future days
            latent_seqs *= valid_mask

            # Compute the mean over the sequence, strictly averaging only the valid days.
            # The clamp(min=1) prevents division by zero in case of an empty window.
            h = latent_seqs.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            # If no mask is provided, perform a simple average pooling over the entire sequence
            # (Typically used during inference on full sequences)
            h = latent_seqs.mean(dim=1)

        # Final projection through a linear layer and non-linearity
        h = self.enc_out_layer(input=h)
        return h
