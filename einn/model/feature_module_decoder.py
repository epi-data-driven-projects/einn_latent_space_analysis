import torch
import torch.nn as nn

from einn.model.base_neural_network import BaseNeuralNetwork


class FeatureModuleDecoder(BaseNeuralNetwork):
    """
    Decoder part of the Feature Module mapping the context vector to latent embeddings.
    """
    def __init__(
            self,
            rnn_out: int = 40,
            dim_out: int = 20,
            n_layers: int = 1,
            bidirectional: bool = True,
            dropout: float = 0.0
    ):
        """
        Initializes the Decoder architecture.

        :param int rnn_out: Hidden size of the RNN layers (matches encoder output).
        :param int dim_out: Final embedding dimension (d_e).
        :param int n_layers: Number of recurrent layers.
        :param bool bidirectional: Whether the Decoder GRU is bidirectional.
        :param float dropout: Dropout probability.
        """
        super().__init__()

        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1

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
            nn.Dropout(p=dropout)
        )

        # Initialize linear layers with Xavier uniform
        self.dec_out_layer.apply(fn=self._init_weights)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Decodes the context vector across specified time steps.
        :param torch.Tensor h: Context vector from encoder. Shape: [Batch, rnn_out].
        :param torch.Tensor t: Scaled time steps. Shape: [Batch, Seq_len, 1].
        :return torch.Tensor: Feature embeddings (e_t^F). Shape: [Batch, Seq_len, dim_out].
        """
        batch_size = h.size(dim=0)
        hidden_size = self.rnn_out // self.num_directions

        # Reshape context vector to match directions: [Batch, num_directions, hidden_size]
        h_reshaped = h.view(batch_size, self.num_directions, hidden_size)

        # Transpose to match GRU's expected h0 shape: [num_layers * num_directions, Batch, hidden_size]
        h0 = h_reshaped.transpose(dim0=0, dim1=1).repeat(self.n_layers, 1, 1).contiguous()

        # Pass through Decoder GRU. Output shape: [Batch, Seq_len, rnn_out]
        latent_seqs, _ = self.dec_rnn(input=t, hx=h0)

        # Final projection to embedding space. Shape: [Batch, Seq_len, dim_out]
        e_t_f = self.dec_out_layer(input=latent_seqs)
        return e_t_f
