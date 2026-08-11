import math
import torch
import torch.nn as nn


class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer-like attention.
    Designed exclusively for batch-first tensors.
    """

    def __init__(self, dim_in: int = 40, value_dim: int = 40, key_dim: int = 40) -> None:
        """
        Initializes the attention layers.
        :param int dim_in: Dimensionality of input sequence features.
        :param int value_dim: Dimension of the value transform.
        :param int key_dim: Dimension of the key transform.
        """

        super().__init__()
        self.value_layer = nn.Linear(in_features=dim_in, out_features=value_dim)
        self.query_layer = nn.Linear(in_features=dim_in, out_features=value_dim)
        self.key_layer = nn.Linear(in_features=dim_in, out_features=key_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for standard self-attention.
        :param torch.Tensor seq: Sequence tensor. Shape: [Batch, Seq_len, dim_in].
        :return torch.Tensor: Attended sequence. Shape: [Batch, Seq_len, key_dim].
        """
        # Linear projections
        # Output shapes: [Batch, Seq_len, value_dim/key_dim]
        value = self.value_layer(seq)
        query = self.query_layer(seq)
        keys = self.key_layer(seq)

        # Scaled dot-product attention
        # query.transpose(1, 2) swaps Seq_len and Hidden_size for matrix multiplication.
        # Shape of transposed query: [Batch, Hidden, Seq_len]
        # weights shape: [Batch, Seq_len, Seq_len]
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, dim=-1)

        # Apply weights to keys.
        # Note: No `.transpose()` is needed here because batch_first=True!
        # [Batch, Seq_len, Seq_len] @ [Batch, Seq_len, Hidden] -> [Batch, Seq_len, Hidden]
        out = weights @ keys
        return out

    def forward_mask(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for masked self-attention, ignoring specific sequence steps.
        :param torch.Tensor seq: Sequence tensor. Shape: [Batch, Seq_len, dim_in].
        :param torch.Tensor mask: Mask tensor. Shape: [Batch, Seq_len].
                                         Values should be 1 for valid steps, 0 for ignored steps.
        :return torch.Tensor: Masked and attended sequence. Shape: [Batch, Seq_len, key_dim].
        """

        value = self.value_layer(seq)
        query = self.query_layer(seq)
        keys = self.key_layer(seq)

        # Compute raw attention scores
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.exp(weights)

        # Apply mask and re-normalize (manual masked softmax)
        # mask.unsqueeze(1) creates shape [Batch, 1, Seq_len] to broadcast across queries
        weights *= mask.unsqueeze(1)
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-9

        # Multiply by keys and apply mask to the final output steps
        # mask.unsqueeze(-1) creates shape [Batch, Seq_len, 1]
        out = (weights @ keys) * mask.unsqueeze(-1)
        return out
