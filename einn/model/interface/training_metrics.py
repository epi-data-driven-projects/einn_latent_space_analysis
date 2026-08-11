from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """
    Dataclass for tracking and logging various loss components during training.
    Useful for generating CSV reports and visualizing training stability.

    Attributes:
        epoch (int): Current epoch number.
        phase (int): Current training phase (1 to 4).
        total_loss (float): The combined, weighted scalar loss used by the optimizer.
        loss_data_T (float): Data matching MSE for the Time module.
        loss_data_F (float): Data matching MSE for the Feature module.
        loss_aux (float): Auxiliary trajectory matching MSE.
        loss_ode_T (float): Physics-informed ODE constraint MSE for the Time module.
        loss_ode_F (float): Physics-informed ODE constraint MSE for the Feature module.
        loss_ode_future (float): Unsupervised physics constraint MSE on future predictions.
        loss_mono (float): Penalty for violating monotonically increasing bounds.
        loss_param (float): Penalty for large jumps in time-varying ODE parameters.
        loss_kd_target (float): Knowledge Distillation MSE on the output states.
        loss_kd_emb (float): Knowledge Distillation MSE on the latent embeddings.
    """
    epoch: int
    phase: int
    total_loss: float = 0.0
    loss_data_T: float = 0.0
    loss_data_F: float = 0.0
    loss_aux: float = 0.0
    loss_ode_T: float = 0.0
    loss_ode_F: float = 0.0
    loss_ode_future: float = 0.0  # Added for future extrapolation tracking
    loss_mono: float = 0.0
    loss_param: float = 0.0
    loss_kd_target: float = 0.0
    loss_kd_emb: float = 0.0
