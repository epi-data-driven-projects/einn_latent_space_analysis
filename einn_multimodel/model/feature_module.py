"""
Shared encoder-decoder FeatureModule with model-label conditioning for multi-model EINN.

Implements the extended Feature branch from thesis Section 2.4 (Figure 10): encodes
observed multivariate time series with two-stage static model-label conditioning and
decodes timestep-level latent embeddings e_t^F without temporal pooling or projection heads.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from einn.model.base_neural_network import BaseNeuralNetwork
from einn.model.feature_module_decoder import FeatureModuleDecoder
from einn_multimodel.model.feature_module_encoder import FeatureModuleEncoder
from einn_multimodel.model.interface.feature_module_outputs import FeatureModuleOutputs


class FeatureModule(BaseNeuralNetwork):
    """
    Encoder-decoder Feature module with static model-label conditioning and optional L2 normalization.

    Maps observed input sequences x of shape (B, N, d_x), query times t of shape (B, T, 1),
    and model labels of shape (B,) to timestep-level latent embeddings e_t^F of shape (B, T, d_e).
    No auxiliary classifier, projection head, or temporal pooling is applied; the exact same
    embeddings feed OutputModule, embedding knowledge distillation, and SupConLoss (Section 2.4).
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        rnn_out: int = 40,
        dim_out: int = 20,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        num_models: int = 2,
        model_label_dim: int = 5,
        normalize_embeddings: bool = False,
        condition_on_model_label: bool = True,
    ) -> None:
        """
        Initialize the conditioned FeatureModuleEncoder, FeatureModuleDecoder, and encoder weights.

        :param int dim_seq_in: Number of input observation features d_x per timestep (default 5).
        :param int rnn_out: Total hidden width of the encoder/decoder GRUs (default 40; Fig. 10).
        :param int dim_out: Output latent embedding dimension d_e (e.g. 20 or 5; Table 1).
        :param int n_layers: Number of stacked GRU layers in both encoder and decoder.
        :param bool bidirectional: If True, uses bidirectional GRUs in encoder and decoder.
        :param float dropout: Dropout probability applied within the encoder and decoder.
        :param int num_models: Number of epidemiological model classes in the label embedding table.
        :param int model_label_dim: Dimension of the static learned model-label vector e_m (default 5).
        :param bool normalize_embeddings: If True, L2-normalizes each timestep embedding e_t^F
            along the feature dimension d_e before returning.
        :param bool condition_on_model_label: If True, conditions the encoder on the sample's
            model label; if False, supplies a single constant token (0) to the encoder as an
            ablation control while keeping both fusion layers intact (thesis Sections 4.2 and 5).
        :return None: Registers encoder and decoder submodules and initializes encoder weights.
        """
        super().__init__()
        self.normalize_embeddings: bool = normalize_embeddings
        self.condition_on_model_label: bool = condition_on_model_label
        self.encoder: FeatureModuleEncoder = FeatureModuleEncoder(
            dim_seq_in=dim_seq_in,
            rnn_out=rnn_out,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            num_models=num_models,
            model_label_dim=model_label_dim,
        )
        self.decoder: FeatureModuleDecoder = FeatureModuleDecoder(
            rnn_out=rnn_out,
            dim_out=dim_out,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.encoder.apply(self._init_weights)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor],
        assumed_model_label: Optional[torch.Tensor] = None,
        true_model_label: Optional[torch.Tensor] = None,
    ) -> FeatureModuleOutputs:
        """
        Encode observed prefixes with model-label conditioning and decode timestep embeddings e_t^F.

        :param torch.Tensor x: Standardized input observation sequences of shape (B, N, d_x).
        :param torch.Tensor t: Min-max scaled query time coordinates of shape (B, T, 1).
        :param Optional[torch.Tensor] mask: Optional padding mask of shape (B, N) where 0 marks
            observed timesteps and 1 marks padded future steps.
        :param Optional[torch.Tensor] assumed_model_label: Backward-compatible alias for
            true_model_label of shape (B,).
        :param Optional[torch.Tensor] true_model_label: Known epidemiological model IDs of
            shape (B,); required when num_models > 1.
        :return FeatureModuleOutputs: Structured output containing embeddings of shape (B, T, d_e)
            and selected_model_label of shape (B,).
        :raises ValueError: If true_model_label and assumed_model_label conflict, if no label is
            provided for a multi-model encoder, or if the label count does not match batch size B.
        """
        # 1. Encode observed prefixes with packed BiGRU: latent_seqs (B, N, rnn_out), valid_mask (B, N).
        latent_seqs, _, valid_mask = self.encoder.encode_sequence(x=x, mask=mask)

        # 2. Resolve and validate the per-trajectory integer model label of shape (B,).
        if true_model_label is not None and assumed_model_label is not None:
            if not torch.equal(true_model_label.reshape(-1).long(), assumed_model_label.reshape(-1).long()):
                raise ValueError("The true-only experiment cannot use conflicting model labels.")
        model_label = true_model_label if true_model_label is not None else assumed_model_label
        if model_label is None:
            if self.encoder.model_embedding.num_embeddings != 1:
                raise ValueError("true_model_label is required for multi-model FeatureModule.")
            selected_label = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            selected_label = model_label.reshape(-1).long().to(x.device)
        if selected_label.numel() != x.size(0):
            raise ValueError("One model label is required per trajectory.")

        # 3. Apply ablation switch: when disabled, feed constant token 0 to the encoder while
        # retaining both fusion layers and preserving selected_label for downstream routing/SupCon.
        conditioning_label = (
            selected_label if self.condition_on_model_label
            else torch.zeros_like(selected_label)
        )

        # 4. Fuse static label embedding before/after self-attention -> context h of shape (B, rnn_out).
        conditioned_context = self.encoder.condition_sequence(
            latent_seqs=latent_seqs,
            valid_mask=valid_mask,
            assumed_model_label=conditioning_label,
        )

        # 5. Decode timestep-level latent embeddings e_t^F of shape (B, T, d_e).
        embeddings = self.decoder(h=conditioned_context, t=t)
        if self.normalize_embeddings:
            # L2-normalize per timestep along d_e (dim=-1), never across the time axis.
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)

        return FeatureModuleOutputs(
            embeddings=embeddings,
            selected_model_label=selected_label,
        )
