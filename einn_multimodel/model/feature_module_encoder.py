from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from einn.model.base_neural_network import BaseNeuralNetwork


class FeatureModuleEncoder(BaseNeuralNetwork):
    """
    Recurrent self-attention encoder with two-stage static model-label conditioning.

    Maps a padded multivariate observation sequence x of shape (B, N, dim_seq_in) and an
    integer model label of shape (B,) into a fixed-size conditioned context vector h of
    shape (B, rnn_out) used to initialize the FeatureModuleDecoder hidden state (Section 2.4, Fig. 10).
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        rnn_out: int = 40,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        num_models: int = 2,
        model_label_dim: int = 5,
    ):
        """
        Initialize the GRU encoder, learnable model-label embedding, fusion layers, and self-attention.

        :param int dim_seq_in: Number of input features d_x per timestep (default 5).
        :param int rnn_out: Total hidden output width of the GRU and final context vector h
            (must be even when bidirectional is True; default 40 = 20 per direction, Fig. 10).
        :param int n_layers: Number of stacked GRU layers.
        :param bool bidirectional: If True, uses a bidirectional GRU.
        :param float dropout: Dropout probability applied in stacked GRU, fusion, attention, and output layers.
        :param int num_models: Number of distinct epidemiological model classes in the embedding table.
        :param int model_label_dim: Dimensionality of the static learned model-label embedding e_m (default 5).
        """
        super().__init__()

        if bidirectional and rnn_out % 2:
            raise ValueError(
                "rnn_out must be even for a bidirectional GRU."
            )

        self.rnn_out: int = rnn_out
        self.n_layers: int = n_layers
        self.bidirectional: bool = bidirectional
        self.num_directions: int = 2 if bidirectional else 1
        self.model_label_dim: int = model_label_dim

        # 1. Temporal encoder: BiGRU mapping (B, N, dim_seq_in) -> (B, N, rnn_out).
        self.enc_rnn = nn.GRU(
            input_size=dim_seq_in,
            hidden_size=rnn_out // self.num_directions,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        # Learnable static covariate embedding table: model ID in {0, ..., num_models - 1} -> (B, model_label_dim).
        fusion_input_dim = rnn_out + model_label_dim
        self.model_embedding = nn.Embedding(
            num_models,
            model_label_dim,
        )

        # 2. Pre-attention fusion block: [h_t || e_m] (B, N, rnn_out + model_label_dim) -> (B, N, rnn_out).
        self.label_fusion_1 = nn.Sequential(
            nn.Linear(fusion_input_dim, rnn_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # 3. Single-head self-attention across observed timesteps: (B, N, rnn_out) -> (B, N, rnn_out).
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=rnn_out,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )

        # 4. Post-attention fusion block: re-injects static e_m after attention (Section 2.4, Fig. 10).
        self.label_fusion_2 = nn.Sequential(
            nn.Linear(fusion_input_dim, rnn_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # 5. Final context projection after masked sum pooling: (B, rnn_out) -> (B, rnn_out).
        self.enc_out_layer = nn.Sequential(
            nn.Linear(rnn_out, rnn_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _valid_mask(
        mask: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert the project padding mask (0 = observed, 1 = padded) into a boolean valid-step mask.

        :param Optional[torch.Tensor] mask: Optional padding mask of shape (B, N), or None if unpadded.
        :param torch.Tensor x: Input sequence tensor of shape (B, N, d_x) supplying batch/time dimensions and device.
        :return torch.Tensor: Boolean tensor of shape (B, N) where True indicates an observed timestep.
        """
        if mask is None:
            return torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )

        # Invert project mask convention: 0 -> True (valid/observed), 1 -> False (padded).
        return ~mask.bool()

    def encode_sequence(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the packed GRU over observed prefixes and return sequence states plus an unconditioned summary.

        Uses pack_padded_sequence so padded future steps do not corrupt recurrent hidden states.

        :param torch.Tensor x: Standardized input sequence of shape (B, N, d_x).
        :param Optional[torch.Tensor] mask: Optional padding mask of shape (B, N) (0 = observed, 1 = padded).
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple (latent_seqs, context_unconditioned,
            valid_mask) with shapes (B, N, rnn_out), (B, rnn_out), and (B, N) respectively.
        """
        valid_mask = self._valid_mask(mask, x)
        lengths = valid_mask.sum(dim=1)  # Shape: (B,)

        if torch.any(lengths == 0):
            raise ValueError(
                "Every sequence must contain at least one unpadded time step."
            )

        # Pack variable-length prefixes so the BiGRU ignores padded future positions.
        packed = pack_padded_sequence(
            x,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.enc_rnn(packed)

        # Restore padded tensor shape (B, N, rnn_out).
        latent_seqs, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )

        # Masked sum pooling over observed steps to produce a label-free baseline context (B, rnn_out).
        weights = valid_mask.unsqueeze(-1).to(latent_seqs.dtype)
        context_unconditioned = (
            (latent_seqs * weights).sum(dim=1)
        )

        return latent_seqs, context_unconditioned, valid_mask

    def condition_sequence(
        self,
        latent_seqs: torch.Tensor,
        valid_mask: torch.Tensor,
        assumed_model_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject the static model-label embedding before and after self-attention, then pool to context h.

        Follows thesis Section 2.4 (Figure 10): concatenates e_m with GRU outputs, applies
        label_fusion_1, runs masked self-attention, re-concatenates e_m, applies label_fusion_2,
        sum-pools over valid_mask, and projects through enc_out_layer.

        :param torch.Tensor latent_seqs: GRU hidden sequence of shape (B, N, rnn_out).
        :param torch.Tensor valid_mask: Boolean observed-step mask of shape (B, N) (True = observed).
        :param torch.Tensor assumed_model_label: Integer model IDs of shape (B,) (or reshapable to (B,)).
        :return torch.Tensor: Model-conditioned context vector h of shape (B, rnn_out).
        """
        if assumed_model_label.ndim != 1:
            assumed_model_label = assumed_model_label.reshape(-1)

        # Embed static model label and broadcast across time: (B,) -> (B, model_label_dim) -> (B, N, model_label_dim).
        label_embedding = self.model_embedding(
            assumed_model_label.long()
        )
        label_sequence = label_embedding.unsqueeze(1).expand(-1, latent_seqs.size(1), -1,)

        # Fusion 1 (pre-attention): (B, N, rnn_out + model_label_dim) -> (B, N, rnn_out).
        fused = torch.cat(
            (latent_seqs, label_sequence),
            dim=-1,
        )
        fused = self.label_fusion_1(fused)

        # Masked self-attention: key_padding_mask uses True to ignore padded positions.
        key_padding_mask = ~valid_mask
        fused, _ = self.attn_layer(
            query=fused,
            key=fused,
            value=fused,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Fusion 2 (post-attention): re-inject e_m so label information shapes post-attention features.
        fused = torch.cat(
            (fused, label_sequence),
            dim=-1,
        )
        fused = self.label_fusion_2(fused)

        # Masked sum pooling over observed timesteps: (B, N, rnn_out) -> (B, rnn_out) (Fig. 10).
        weights = valid_mask.unsqueeze(-1).to(fused.dtype)
        context = (
            (fused * weights).sum(dim=1)
        )

        return self.enc_out_layer(context)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        assumed_model_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observed time series into both model-conditioned and unconditioned context vectors.

        :param torch.Tensor x: Standardized input observation sequence of shape (B, N, d_x).
        :param Optional[torch.Tensor] mask: Optional padding mask of shape (B, N) (0 = observed, 1 = padded).
        :param torch.Tensor assumed_model_label: Integer model IDs of shape (B,).
        :return tuple[torch.Tensor, torch.Tensor]: Pair (conditioned, unconditioned) of context
            vectors, each of shape (B, rnn_out).
        """
        latent_seqs, unconditioned, valid_mask = (
            self.encode_sequence(x=x, mask=mask)
        )

        conditioned = self.condition_sequence(
            latent_seqs=latent_seqs,
            valid_mask=valid_mask,
            assumed_model_label=assumed_model_label,
        )

        return conditioned, unconditioned
