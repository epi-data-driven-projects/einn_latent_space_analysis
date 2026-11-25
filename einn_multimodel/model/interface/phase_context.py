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
