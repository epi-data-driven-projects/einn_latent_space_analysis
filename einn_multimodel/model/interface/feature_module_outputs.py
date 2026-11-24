"""
Dataclass container for FeatureModule forward outputs in multi-model EINN.
"""
from dataclasses import dataclass

import torch


@dataclass
class FeatureModuleOutputs:
    """
    Timestep-level Feature-branch embeddings and the model label used for routing/conditioning.

    Represents the shared latent output e_t^F of shape (B, T, d_e) produced by FeatureModule
    (thesis Section 2.4, Figure 10), optionally L2-normalized per timestep along d_e and
    passed directly to OutputModule, knowledge distillation, and SupConLoss without a
    separate projection head.

    :param torch.Tensor embeddings: Feature-branch latent tensor e_t^F of shape (B, T, d_e).
    :param torch.Tensor selected_model_label: Integer model IDs of shape (B,) associated with
        each sequence in the batch.
    """

    embeddings: torch.Tensor             # Shape: (B, T, d_e)
    selected_model_label: torch.Tensor   # Shape: (B,)
