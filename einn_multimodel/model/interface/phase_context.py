from dataclasses import dataclass

import torch

from einn_multimodel.model.interface.einn_models import EINNModels


@dataclass
class PhaseContext:
    """Batch data and model labels required by one EINN optimization step.

    :param phase_num: Active optimization phase number, ranging from one through four.
    :param epoch: Index of the current optimization epoch.
    :param x: Padded input observations and synthetic covariates for the batch.
    :param y: Target epidemiological time series in its training-scaled representation.
    :param t: Time coordinates of every requested observation or extrapolation step.
    :param aux_targets: Reference mechanistic compartment trajectories for auxiliary supervision.
    :param models: Bundle containing all neural and epidemiological model modules.
    :param mask: Observation mask with zero for observed time points and one for padding.
    :param true_model_label: Integer identifier of the true synthetic data-generating system.
    :param assumed_model_label: Integer identifier of the hypothesized mechanistic model.
    :param is_mismatch: Flag indicating a disagreement between true and assumed model labels.
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
        """Validate consistency of the initialized dataclass fields.


        :return None: No return value; state or serialized artifacts are updated in place.
        """
        batch_size = self.x.shape[0]
        for name in ("true_model_label", "assumed_model_label", "is_mismatch"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != batch_size:
                raise ValueError(f"{name} batch dimension does not match x.")
        if len(self.models.time_modules) > 1 and self.assumed_model_label is None:
            raise ValueError("assumed_model_label is required for a multi-model batch.")
