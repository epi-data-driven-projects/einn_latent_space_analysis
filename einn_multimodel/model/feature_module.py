from typing import Optional

import torch
import torch.nn.functional as F

from einn.model.base_neural_network import BaseNeuralNetwork
from einn.model.feature_module_decoder import FeatureModuleDecoder
from einn_multimodel.model.feature_module_encoder import FeatureModuleEncoder
from einn_multimodel.model.interface.feature_module_outputs import FeatureModuleOutputs


class FeatureModule(BaseNeuralNetwork):
    """Original EINN Feature encoder/decoder with true-model conditioning.

    The true label enters before and after self-attention in the encoder.
    No classifier, projection head, or temporal embedding pooling is used.
    Optional L2 normalization is applied per timestep to decoder embeddings.
    The exact same embeddings feed OutputModule, embedding KD, and SupCon.
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
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.condition_on_model_label = condition_on_model_label
        self.encoder = FeatureModuleEncoder(
            dim_seq_in=dim_seq_in,
            rnn_out=rnn_out,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            num_models=num_models,
            model_label_dim=model_label_dim,
        )
        self.decoder = FeatureModuleDecoder(
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
        latent_seqs, _, valid_mask = self.encoder.encode_sequence(x=x, mask=mask)
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

        # Control experiment: same learned token for both classes; no class id
        # directly enters the encoder. The encoder still has both fusion layers.
        conditioning_label = (
            selected_label if self.condition_on_model_label
            else torch.zeros_like(selected_label)
        )
        conditioned_context = self.encoder.condition_sequence(
            latent_seqs=latent_seqs,
            valid_mask=valid_mask,
            assumed_model_label=conditioning_label,
        )
        embeddings = self.decoder(h=conditioned_context, t=t)
        if self.normalize_embeddings:
            # Normalize along d_e, NOT along time. No temporal pooling.
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        return FeatureModuleOutputs(
            embeddings=embeddings,
            selected_model_label=selected_label,
        )
