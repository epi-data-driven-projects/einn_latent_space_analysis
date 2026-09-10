from dataclasses import dataclass
import torch

from einn.model.interface.einn_models import EINNModels


@dataclass
class PhaseContext:
    """
    Data class representing the current state of the training phase.

    :param int phase_num: current optimization phase identifier (1, 2, 3, or 4)
    :param int epoch: current training epoch number
    :param torch.Tensor X: tensor of noisy input observations for the current batch
    :param torch.Tensor y: tensor of target variables
    :param torch.Tensor t: tensor of time steps to evaluate
    :param torch.Tensor aux_targets: tensor of ideal trajectories for the initial auxiliary loss
    :param EINNModels models: container holding the initialized neural and ODE models
    """
    phase_num: int
    epoch: int
    X: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    aux_targets: torch.Tensor
    models: EINNModels
