from dataclasses import dataclass
import torch

from einn.model.interface.einn_models import EINNModels


@dataclass
class PhaseContext:
    """
    Data class representing the current state of the training phase.

    :param phase_num: current optimization phase identifier (1, 2, 3, or 4)
    :param epoch: current training epoch number
    :param x: tensor of noisy input observations for the current batch
    :param y: tensor of target variables
    :param t: tensor of time steps to evaluate
    :param aux_targets: tensor of ideal trajectories for the initial auxiliary loss
    :param models: container holding the initialized neural and ODE models
    :param mask: mask holding the current training phase mask
    """
    phase_num: int
    epoch: int
    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    aux_targets: torch.Tensor
    models: EINNModels
    mask: torch.Tensor | None = None
