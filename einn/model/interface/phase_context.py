from dataclasses import dataclass

import torch

from einn.model.interface.einn_models import EINNModels


@dataclass
class PhaseContext:
    """
    Context object carrying the state and data for a single training step (batch).
    This cleanly packages all necessary variables to avoid bloated method signatures
    in the phase optimizers and loss calculations.

    - phase_num (int): The current training phase (1, 2, 3, or 4).
    - epoch (int): The current training epoch number.
    - X (torch.Tensor): Observed noisy input features.
                      Expected Shape: [Batch, Seq_len, d_x].
    - y (torch.Tensor): Target variable (e.g., mortality or infected).
                      Expected Shape: [Batch, Seq_len, 1].
    - t (torch.Tensor): The scaled time vector.
                      Expected Shape: [Batch, Seq_len, 1].
    - aux_targets (torch.Tensor): Ideal trajectories for early-phase pre-calibration.
                                Expected Shape: [Batch, Seq_len, d_s].
    - models (EINNModels): Container holding references to the active
                         neural and physical ODE models.
    """
    phase_num: int
    epoch: int
    X: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    aux_targets: torch.Tensor
    models: EINNModels
