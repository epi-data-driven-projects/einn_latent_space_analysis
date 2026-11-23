from dataclasses import dataclass

import torch


@dataclass
class FeatureModuleOutputs:
    """Unprojected EINN Feature embeddings, optionally L2-normalized per t.

    :param embeddings: Feature-module latent tensor with shape [batch, time, latent_dim].
    :param selected_model_label: Model identifier used to condition the Feature encoder.
    """

    embeddings: torch.Tensor             # [B, T, d_e]
    selected_model_label: torch.Tensor   # [B]
