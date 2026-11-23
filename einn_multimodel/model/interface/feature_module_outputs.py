from dataclasses import dataclass

import torch


@dataclass
class FeatureModuleOutputs:
    """Unprojected EINN Feature embeddings, optionally L2-normalized per t."""

    embeddings: torch.Tensor             # [B, T, d_e]
    selected_model_label: torch.Tensor   # [B]
