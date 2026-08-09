from dataclasses import dataclass
import torch
from einn.model.interface.einn_models import EINNModels


@dataclass
class PhaseContext:
    phase_num: int
    epoch: int
    X: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    aux_targets: torch.Tensor
    models: EINNModels
