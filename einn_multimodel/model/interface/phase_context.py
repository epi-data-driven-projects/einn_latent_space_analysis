from dataclasses import dataclass

import torch

from einn_multimodel.model.interface.einn_models import EINNModels


@dataclass
class PhaseContext:
    """Batch data and model labels required by one EINN optimization step."""

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
        batch_size = self.x.shape[0]
        for name in ("true_model_label", "assumed_model_label", "is_mismatch"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != batch_size:
                raise ValueError(f"{name} batch dimension does not match x.")
        if len(self.models.time_modules) > 1 and self.assumed_model_label is None:
            raise ValueError("assumed_model_label is required for a multi-model batch.")
