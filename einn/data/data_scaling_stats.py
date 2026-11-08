from dataclasses import dataclass
from typing import List
import torch

@dataclass
class DataScalingStats:
    """
    Data class encapsulating runtime scaling statistics and initial conditions.

    :param torch.Tensor aux_mean: The mean values of the auxiliary/compartment data used for Z-score normalization.
    :param torch.Tensor aux_std: The standard deviation values of the auxiliary data used for Z-score normalization.
    :param torch.Tensor t_min: The minimum time value of the sequence, used for Min-Max scaling of the temporal domain.
    :param torch.Tensor t_max: The maximum time value of the sequence, used for Min-Max scaling of the temporal domain.
    :param List[float] initial_states: The normalized initial state values of the ODE compartments
                                           (e.g., S, E, I, R, M) at t=0.
    """
    aux_mean: torch.Tensor
    aux_std: torch.Tensor
    t_min: torch.Tensor
    t_max: torch.Tensor
    initial_states: List[float]
