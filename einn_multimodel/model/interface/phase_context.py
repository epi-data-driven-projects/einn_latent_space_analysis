"""
Dataclass encapsulating batch tensors, model references, and labels for one training step.
"""
from dataclasses import dataclass

import torch

from einn_multimodel.model.interface.einn_models import EINNModels


@dataclass
class PhaseContext:
    """
    Batch inputs, active phase metadata, and model labels for one multi-model EINN step.

    Passed through EINNForwardEngine and EINNLoss during the four-phase optimization loop
    (thesis Sections 2.2 and 2.4) and optional mismatched-model control runs (Section 5).

    :param int phase_num: Active training phase index in {1, 2, 3, 4} (thesis Section 2.2).
    :param int epoch: Current optimization epoch or inner iteration index.
    :param torch.Tensor x: Standardized, zero-padded input feature sequences of shape (B, T, d_x).
    :param torch.Tensor y: Standardized ground-truth target series of shape (B, T, 1).
    :param torch.Tensor t: Unscaled time coordinate grids of shape (B, T, 1).
    :param torch.Tensor aux_targets: Standardized auxiliary reference compartment states
        of shape (B, T, d_s).
    :param EINNModels models: Container holding all per-model Time/ODE modules and shared
        Feature/Output modules.
    :param torch.Tensor | None mask: Optional padding mask of shape (B, T) where 0 marks
        observed timesteps and 1 marks padded future timesteps.
    :param torch.Tensor | None true_model_label: Ground-truth generating model IDs a_i
        of shape (B,); used by SupConLoss and mismatch masking.
    :param torch.Tensor | None assumed_model_label: Assumed epidemiological model IDs of
        shape (B,) used to route active TimeModules and ODE models; required when
        len(models.time_modules) > 1.
    :param torch.Tensor | None is_mismatch: Optional boolean mask of shape (B,) indicating
        samples where true_model_label != assumed_model_label.
    """

    phase_num: int
    epoch: int
    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    aux_targets: torch.Tensor
    models: EINNModels
    mask: torch.Tensor | None = None
    true_model_label: torch.Tensor | None = None
    assumed_model_label: torch.Tensor | None = None
    is_mismatch: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """
        Validate batch-dimension alignment of label tensors and multi-model routing requirements.

        :return None: Validates dataclass invariants in place.
        :raises ValueError: If any label/mismatch tensor's leading dimension differs from B,
            or if assumed_model_label is None in a multi-model setup.
        """
        batch_size = self.x.shape[0]
        for name in ("true_model_label", "assumed_model_label", "is_mismatch"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != batch_size:
                raise ValueError(f"{name} batch dimension does not match x.")
        if len(self.models.time_modules) > 1 and self.assumed_model_label is None:
            raise ValueError("assumed_model_label is required for a multi-model batch.")
