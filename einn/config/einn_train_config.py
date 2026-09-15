from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class EINNTrainConfig:
    """
    Configuration data class for the EINN training process and loss calculations.

    :param float learning_rate: learning rate for optimizers
    :param int epochs_per_phase: number of training epochs per phase
    :param int future_steps: number of steps for future prediction
    :param float early_stopping_min_delta: minimum delta for early stopping
    :param int early_stopping_patience: patience for early stopping
    :param bool early_stopping_percentage: whether to use percentage for delta
    :param str device: target device for tensors
    :param Dict[str, float] loss_weights: dictionary containing weights for various loss components
    """
    learning_rate: float = 0.001
    epochs_per_phase: int = 1000
    future_steps: int = 30

    early_stopping_min_delta: float = 0.0
    early_stopping_patience: int = 10
    early_stopping_percentage: bool = False

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'data_T': 1.0,
        'data_F': 1.0,
        'aux': 0.1,
        'ode_T': 10.0,
        'ode_F': 10.0,
        'ode_future_T': 10.0,
        'ode_future_F': 10.0,
        'mono': 1.0,
        'param': 0.001,
        'kd_target': 1.0,
        'kd_emb': 5.0
    })
