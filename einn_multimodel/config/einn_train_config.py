from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class EINNTrainConfig:
    """Configuration for the four-phase EINN training procedure.

    :param learning_rate: The learning rate used by this operation.
    :param future_steps: The future steps used by this operation.
    :param reps: The reps used by this operation.
    :param early_stopping_min_delta: The early stopping min delta used by this operation.
    :param early_stopping_patience: The early stopping patience used by this operation.
    :param early_stopping_percentage: The early stopping percentage used by this operation.
    :param device: PyTorch device used for computations.
    :param loss_weights: The loss weights used by this operation.
    :param supcon_temperature: The supcon temperature used by this operation.
    :param mask_mismatched_losses: The mask mismatched losses used by this operation.
    """

    learning_rate: float = 0.001
    future_steps: int = 30
    reps: Dict[str, int] = field(default_factory=lambda: {
        "phase_1": 300,
        "phase_2": 300,
        "phase_3": 300,
        "phase_4": 300,
    })
    early_stopping_min_delta: float = 0.0
    early_stopping_patience: int = 10
    early_stopping_percentage: bool = False
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "data_T": 1.0,
        "data_F": 1.0,
        "aux": 0.1,
        "ode_T": 10.0,
        "ode_F": 10.0,
        "ode_future_T": 10.0,
        "ode_future_F": 10.0,
        "mono": 1.0,
        "param": 0.001,
        "kd_target": 1.0,
        "kd_emb": 5.0,
        "ic": 10.0,
        "ic_F": 10.0,
        "supcon": 0.0,
    })
    supcon_temperature: float = 0.1
    # If enabled, physics/Time/KD objectives ignore samples whose true
    # model differs from the assumed model. Feature data/IC/SupCon stay active.
    mask_mismatched_losses: bool = False

    def __post_init__(self) -> None:
        """Validate consistency of the initialized dataclass fields.


        :return None: No return value; state or serialized artifacts are updated in place.
        """
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.supcon_temperature <= 0:
            raise ValueError("supcon_temperature must be positive.")
        if not isinstance(self.mask_mismatched_losses, bool):
            raise TypeError("mask_mismatched_losses must be a bool.")
