from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """
    Class for tracking training metrics
    """
    epoch: int
    phase: int
    total_loss: float = 0.0
    loss_data_T: float = 0.0
    loss_data_F: float = 0.0
    loss_aux: float = 0.0
    loss_ode_T: float = 0.0  # ide lehet kell majd a "future"-ös
    loss_ode_F: float = 0.0  # ide lehet kell majd a "future"-ös
    loss_mono: float = 0.0
    loss_param: float = 0.0
    loss_kd_target: float = 0.0
    loss_kd_emb: float = 0.0
