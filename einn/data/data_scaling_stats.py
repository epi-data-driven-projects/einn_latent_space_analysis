from dataclasses import dataclass
from typing import List
import torch

@dataclass
class DataScalingStats:
    """
    Futásidőben (a preprocesszálás során) kiszámolt skálázási statisztikák és 
    kezdeti feltételek, amelyek a visszaskálázáshoz és az ODE veszteségszámításhoz szükségesek.
    """
    aux_mean: torch.Tensor
    aux_std: torch.Tensor
    t_min: torch.Tensor
    t_max: torch.Tensor
    initial_states: List[float]
