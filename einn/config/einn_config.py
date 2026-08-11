from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class EINNConfig:
    """
    Configuration dataclass for the EINN model.
    Stores hyperparameters, model dimensions, and training settings.
    """
    # TODO: param docstrings!!!
    # Model dimensions
    d_x: int = 10  # Number of input features
    d_e: int = 20  # Dimension of the latent embedding
    d_s: int = 5  # Number of ODE compartment states (e.g., 5 for SEIRM)
    d_p: int = 4  # Number of ODE parameters

    # Training hyperparameters
    learning_rate: float = 0.001
    epochs_per_phase: int = 100

    # Future prediction settings
    future_steps: int = 30  # Number of time steps to forecast into the future

    # Device configuration (Automatically uses GPU if available, otherwise CPU)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loss weighting factors for different phases and objectives
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'data_T': 1.0,
        'data_F': 1.0,
        'aux': 0.1,
        'ode_T': 10.0,
        'ode_F': 10.0,
        'mono': 1.0,
        'param': 0.001,
        'kd_target': 1.0,
        'kd_emb': 5.0,
        'ode_future': 10.0
    })
