from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    epoch: int
    phase: int
    rep: int

    total_loss: float
    loss_data_T: float = 0.0
    loss_data_F: float = 0.0
    loss_aux: float = 0.0
    loss_ode_T: float = 0.0
    loss_ode_F: float = 0.0
    loss_ode_future_T: float = 0.0
    loss_ode_future_F: float = 0.0
    loss_mono: float = 0.0
    loss_param: float = 0.0
    loss_kd_target: float = 0.0
    loss_kd_emb: float = 0.0
# TODO: a long docstring explaining all the variables
