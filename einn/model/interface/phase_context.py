from dataclasses import dataclass

import torch

from einn.model.interface.einn_models import EINNModels


@dataclass
class PhaseContext:
    """Batch data and metadata used by one EINN optimization step."""

    phase_num: int
    epoch: int
    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    aux_targets: torch.Tensor
    models: EINNModels
    mask: torch.Tensor | None = None          # input mask: 1 = valid
    target_mask: torch.Tensor | None = None   # target mask: 1 = observed
    history_steps: int | None = None          # overlap length before forecast horizon
