from typing import Optional, Dict, List

import torch
import torch.nn as nn

from einn.config.einn_config import EINNConfig
from einn.ode.base_ode_model import BaseODEModel
from einn.model.interface.phase_context import PhaseContext
from einn.model.interface.network_outputs import NetworkOutputs


class EINNLoss(nn.Module):
    """
    Calculates the comprehensive loss for the EINN framework, including data matching,
    physics-informed ODE constraints, knowledge distillation (KD), monotonicity,
    parameter smoothness, and future extrapolation constraints.
    """

    def __init__(self, config: EINNConfig, ode_model: BaseODEModel, scalers: dict,
                 target_state_index: int = -1, mono_indices: Optional[List[int]] = None):
        """
        Initializes the loss module with all penalty weights and configurations.

        Args:
            config (EINNConfig): Configuration containing the loss weights.
            ode_model (BaseODEModel): The physical ODE model (e.g., SEIRM, SIR).
            scalers (dict): Dictionary of data scalers used for inverse scaling.
            target_state_index (int): Index of the predicted state corresponding to the target data 'y'.
                                      Defaults to -1 (last state).
            mono_indices (List[int]): Indices of the states that must strictly be monotonically
                                      increasing (e.g., Recovered or Dead compartments).
                                      Defaults to None. For SEIRM, it's usually [3, 4].
        """
        super(EINNLoss, self).__init__()
        self.weights: Dict[str, float] = config.loss_weights
        self.ode_model = ode_model
        self.scalers = scalers
        self.target_state_index = target_state_index
        self.mono_indices = mono_indices if mono_indices is not None else []
        self.mse = nn.MSELoss()

    def calc_data_loss(self, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error (MSE) between the predicted target state and actual observations.
        Shapes: states [Batch, Seq_len, d_s], targets [Batch, Seq_len, 1].
        """
        pred_target = states[..., self.target_state_index: self.target_state_index + 1]
        return self.mse(pred_target, targets)

    def calc_ode_loss(self, ds_dt_nn: torch.Tensor, states: torch.Tensor, detach_params: bool) -> torch.Tensor:
        """
        MSE between the neural network's gradient and the ODE's analytical gradient.
        Returns the scalar loss and the retrieved physical parameters for smoothness penalty.
        """
        ode_solution = self.ode_model.get_derivatives(states, detach_params=detach_params)
        return self.mse(ds_dt_nn, ode_solution.ds_dt), ode_solution.params

    def calc_monotonicity_loss(self, ds_dt: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric ReLU penalty. Punishes the network if compartments that should only
        grow (like Mortality or Recovered) have a negative time derivative.

        Args:
            ds_dt (torch.Tensor): Neural network derivative [Batch, Seq_len, d_s].
        """
        if not self.mono_indices:
            return torch.tensor(0.0, device=ds_dt.device)

        # Select the derivatives of monotonic compartments
        mono_derivatives = ds_dt[..., self.mono_indices]

        # ReLU(-x) is only > 0 when x is negative. We penalize negative derivatives.
        return torch.mean(torch.relu(-mono_derivatives))

    def calc_parameter_smoothness_loss(self, params: torch.Tensor) -> torch.Tensor:
        """
        Minimizes large jumps in time-varying ODE parameters between consecutive time steps.

        Args:
            params (torch.Tensor): Time-dependent ODE parameters [Batch, Seq_len, d_p].
        """
        if params.size(1) < 2:  # Cannot compute smoothness on a single time step
            return torch.tensor(0.0, device=params.device)

        # Mean squared difference between t and t-1
        return torch.mean((params[:, 1:, :] - params[:, :-1, :]) ** 2)

    def calc_knowledge_distillation_target_loss(self, s_t: torch.Tensor, s_t_f: torch.Tensor) -> torch.Tensor:
        """KD loss for aligning the physical compartment states of the two modules."""
        return self.mse(s_t, s_t_f)

    def calc_knowledge_distillation_emb_loss(self, e_t: torch.Tensor, e_t_f: torch.Tensor) -> torch.Tensor:
        """KD loss for aligning the latent embeddings of the two modules."""
        return self.mse(e_t, e_t_f)

    def forward(self, context: PhaseContext, outputs: NetworkOutputs,
                ds_dt_t: Optional[torch.Tensor] = None, ds_dt_f: Optional[torch.Tensor] = None,
                ds_dt_future: Optional[torch.Tensor] = None, states_future: Optional[torch.Tensor] = None,
                detach_ode_params: bool = False) -> torch.Tensor:
        """
        Aggregates all active loss components based on the current training phase.

        Args:
            context (PhaseContext): Holds phase, epoch, and batch data.
            outputs (NetworkOutputs): Neural module outputs (states and embeddings).
            ds_dt_t, ds_dt_f (torch.Tensor): Calculated gradients for Time and Feature modules.
            ds_dt_future, states_future (torch.Tensor): Future predictions for physics regularization.
            detach_ode_params (bool): Freezes ODE parameters during loss calculation.
        """
        # Ensure loss is on the active device
        total_loss = torch.tensor(0.0, device=context.X.device)

        # Extract weights with safe fallbacks if not defined in config
        w = self.weights

        # ---------------------------------------------------------
        # PHASE 1 & 2: Feature Module Losses
        # ---------------------------------------------------------
        if context.phase_num in [1, 2]:
            loss_data_F = self.calc_data_loss(outputs.s_t_F, context.y)
            total_loss += w.get('data_F', 1.0) * loss_data_F

        if ds_dt_f is not None:
            loss_ode_F, ode_params_F = self.calc_ode_loss(ds_dt_f, outputs.s_t_F, detach_params=detach_ode_params)
            total_loss += w.get('ode_F', 10.0) * loss_ode_F

            # Additional Physics Penalties for Feature Module
            total_loss += w.get('mono', 1.0) * self.calc_monotonicity_loss(ds_dt_f)
            total_loss += w.get('param', 0.001) * self.calc_parameter_smoothness_loss(ode_params_F)

        # ---------------------------------------------------------
        # PHASE 3 & 4: Time Module Losses & Knowledge Distillation
        # ---------------------------------------------------------
        if context.phase_num in [3, 4]:
            loss_data_T = self.calc_data_loss(outputs.s_t, context.y)
            loss_kd_target = self.calc_knowledge_distillation_target_loss(outputs.s_t, outputs.s_t_F)
            loss_kd_emb = self.calc_knowledge_distillation_emb_loss(outputs.e_t, outputs.e_t_F)

            total_loss += w.get('data_T', 1.0) * loss_data_T
            total_loss += w.get('kd_target', 1.0) * loss_kd_target
            total_loss += w.get('kd_emb', 5.0) * loss_kd_emb

        if ds_dt_t is not None:
            loss_ode_T, ode_params_T = self.calc_ode_loss(ds_dt_t, outputs.s_t, detach_params=detach_ode_params)
            total_loss += w.get('ode_T', 10.0) * loss_ode_T

            # Additional Physics Penalties for Time Module
            total_loss += w.get('mono', 1.0) * self.calc_monotonicity_loss(ds_dt_t)
            total_loss += w.get('param', 0.001) * self.calc_parameter_smoothness_loss(ode_params_T)

        # ---------------------------------------------------------
        # FUTURE PHYSICS REGULARIZATION (Unsupervised Forecasting)
        # ---------------------------------------------------------
        if ds_dt_future is not None and states_future is not None:
            # Enforce ODE physics on future predictions where no data is available
            loss_ode_future, _ = self.calc_ode_loss(ds_dt_future, states_future, detach_params=detach_ode_params)
            total_loss += w.get('ode_future', w.get('ode_T', 10.0)) * loss_ode_future

        return total_loss
