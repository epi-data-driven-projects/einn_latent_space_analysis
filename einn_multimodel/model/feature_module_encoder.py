
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from einn.model.base_neural_network import BaseNeuralNetwork


class FeatureModuleEncoder(BaseNeuralNetwork):
    """GRU encoder with model-label conditioning at two fusion points."""

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
        super().__init__()

        if bidirectional and rnn_out % 2:
            raise ValueError(
                "rnn_out must be even for a bidirectional GRU."
            )

        self.rnn_out = rnn_out
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.model_label_dim = model_label_dim

        # 1. GRU: dim_seq_in -> rnn_out
        self.enc_rnn = nn.GRU(
            input_size=dim_seq_in,
            hidden_size=rnn_out // self.num_directions,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        # Mindkét fúzió: (rnn_out + model_label_dim) -> rnn_out
        fusion_input_dim = rnn_out + model_label_dim

        self.model_embedding = nn.Embedding(
            num_models,
            model_label_dim,
        )

        # 2. Első modellcímke-fúzió: 45 -> 40
        self.label_fusion_1 = nn.Sequential(
            nn.Linear(fusion_input_dim, rnn_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # 3. Self-Attention: 40 -> 40
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=rnn_out,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )

        # 4. Második modellcímke-fúzió: 45 -> 40
        self.label_fusion_2 = nn.Sequential(
            nn.Linear(fusion_input_dim, rnn_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # 5. Kontextusvektor végső transzformációja: 40 -> 40
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
        if mask is None:
            return torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )

        # 0 = megfigyelt időpont, 1 = padding
        return ~mask.bool()

    def encode_sequence(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return GRU sequence, label-free context, and valid mask."""

        valid_mask = self._valid_mask(mask, x)
        lengths = valid_mask.sum(dim=1)

        if torch.any(lengths == 0):
            raise ValueError(
                "Every sequence must contain at least one unpadded time step."
            )

        packed = pack_padded_sequence(
            x,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.enc_rnn(packed)

        latent_seqs, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )

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
        """Inject the model-label embedding before and after attention."""

        if assumed_model_label.ndim != 1:
            assumed_model_label = assumed_model_label.reshape(-1)

        # [B] -> [B, 5] -> [B, N, 5]
        label_embedding = self.model_embedding(
            assumed_model_label.long()
        )

        label_sequence = label_embedding.unsqueeze(1).expand(
            -1,
            latent_seqs.size(1),
            -1,
        )

        # Első fúzió:
        # [B, N, 40] + [B, N, 5] -> [B, N, 45]
        fused = torch.cat(
            (latent_seqs, label_sequence),
            dim=-1,
        )

        # [B, N, 45] -> [B, N, 40]
        fused = self.label_fusion_1(fused)

        # Self-Attention: [B, N, 40] -> [B, N, 40]
        key_padding_mask = ~valid_mask

        fused, _ = self.attn_layer(
            query=fused,
            key=fused,
            value=fused,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Második fúzió:
        # [B, N, 40] + [B, N, 5] -> [B, N, 45]
        fused = torch.cat(
            (fused, label_sequence),
            dim=-1,
        )

        # [B, N, 45] -> [B, N, 40]
        fused = self.label_fusion_2(fused)

        # Maskolt átlagpooling: [B, N, 40] -> [B, 40]
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

        latent_seqs, unconditioned, valid_mask = (
            self.encode_sequence(x=x, mask=mask)
        )

        conditioned = self.condition_sequence(
            latent_seqs=latent_seqs,
            valid_mask=valid_mask,
            assumed_model_label=assumed_model_label,
        )

        return conditioned, unconditioned