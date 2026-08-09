from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EINNConfig:
    d_x: int = 10
    d_e: int = 20
    d_s: int = 5
    d_p: int = 4
    learning_rate: float = 0.001
    epochs_per_phase: int = 100
    # TODO: hozzá kellene adni a future-ösöket is
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'data_T': 1.0,
        'data_F': 1.0,
        'aux': 0.1,
        'ode_T': 10.0,
        'ode_F': 10.0,
        'mono': 1.0,
        'param': 0.001,
        'kd_target': 1.0,
        'kd_emb': 5.0
    })
