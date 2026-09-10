import torch
import torch.nn as nn
import torch.nn.functional as F

from einn.config.einn_config import EINNConfig
from einn.ode.base_ode_model import BaseODEModel
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext


class EINNLoss(nn.Module):
    """
    Computes the composite loss for the EINN architecture.
    """

    def __init__(self, config: EINNConfig, ode_model: BaseODEModel, model_type: str):
        """
        Initializes the EINNLoss module.

        :param EINNConfig config: The configuration object containing loss weights.
        :param BaseODEModel ode_model: The epidemiological ODE model.
        :param str model_type: Type of ODE model ('SIR' or 'SEIRM') to determine monotonicity rules.
        """
        super().__init__()
        self.config = config
        self.ode_model = ode_model
        self.weights = config.loss_weights

        # Determine strict monotonicity indices based on the original model logic.
        # Susceptible (S) is always index 0 and must monotonically decrease.
        self.mono_dec_indices = [0]

        # Recovered (R) and Mortality (M) must monotonically increase.
        if model_type == "SIR":
            self.mono_inc_indices = [2]
        elif model_type == "SEIRM":
            self.mono_inc_indices = [3, 4]
        else:
            raise ValueError(f"Unsupported model_type for monotonicity: {model_type}")

    @staticmethod
    def calc_data_loss(states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Mean Squared Error (MSE) between predicted states and observed data.
        Assumes targets map to the first dim(targets) compartments (e.g., Infected).

        :param torch.Tensor states: Predicted compartment states. Shape: [Batch, Seq_len, d_s].
        :param torch.Tensor targets: Ground truth observations. Shape: [Batch, Seq_len, target_dim].
        :return torch.Tensor: Data MSE loss.
        """
        target_dim = targets.shape[2]
        return F.mse_loss(input=states[:, :, :target_dim], target=targets)

    @staticmethod
    def calc_aux_loss(states: torch.Tensor, aux_targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the auxiliary loss against ideal ODE trajectories for Phase 1.

        :param torch.Tensor states: Predicted compartment states. Shape: [Batch, Seq_len, d_s].
        :param torch.Tensor aux_targets: Ideal simulated compartment states.
        :return torch.Tensor: Auxiliary MSE loss.
        """
        return F.mse_loss(input=states, target=aux_targets)

    @staticmethod
    def calc_ode_loss(ds_dt_nn: torch.Tensor, ds_dt_ode: torch.Tensor) -> torch.Tensor:
        """
        Calculates the physics-informed loss for either the Time or Feature module.
        Matches the empirical derivative to the analytical ODE derivative.

        :param torch.Tensor ds_dt_nn: Empirical time derivative from the neural network.
        :param torch.Tensor ds_dt_ode: Analytical derivative from the ODE model.
        :return torch.Tensor: Physics MSE loss.
        """
        return F.mse_loss(input=ds_dt_nn, target=ds_dt_ode)

    @staticmethod
    def calc_monotonicity_loss(
            dS_dt: torch.Tensor, inc_indices: list, dec_indices: list
    ) -> torch.Tensor:
        """
        Applies a squared asymmetric ReLU penalty to enforce monotonic compartments.
        Penalizes when decreasing states (S) have positive derivatives, and when
        increasing states (R, M) have negative derivatives.

        :param torch.Tensor dS_dt: Analytical or empirical time derivative. Shape: [Batch, Seq_len, d_s].
        :param list inc_indices: List of indices that must increase.
        :param list dec_indices: List of indices that must decrease.
        :return torch.Tensor: Monotonicity penalty scalar.
        """
        penalty = torch.tensor(data=0.0, dtype=torch.float32, device=dS_dt.device)

        # Increasing compartments (e.g., R, M): penalty if derivative is negative
        for idx in inc_indices:
            val = -dS_dt[:, :, idx]
            penalty += torch.mean(input=(val * torch.relu(input=val)) ** 2)

        # Decreasing compartments (e.g., S): penalty if derivative is positive
        for idx in dec_indices:
            val = dS_dt[:, :, idx]
            penalty += torch.mean(input=(val * torch.relu(input=val)) ** 2)

        return penalty

    @staticmethod
    def calc_parameter_smoothness_loss(params: torch.Tensor) -> torch.Tensor:
        """
        Penalizes large jumps in time-dependent physical parameters between consecutive steps.

        :param torch.Tensor params: Time-dependent parameter tensor. Shape: [Batch, Seq_len, d_p].
        :return torch.Tensor: Parameter smoothness loss.
        """
        if params.shape[1] <= 1:
            return torch.tensor(data=0.0, dtype=torch.float32, device=params.device)

        diff = params[:, 1:, :] - params[:, :-1, :]
        return torch.mean(input=diff ** 2)

    @staticmethod
    def calc_knowledge_distillation_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Calculates the knowledge distillation discrepancy (MSE) between the embeddings (e_t, e_t^F) or between
        target and predicted states (s_t, s_t^F)

        :param torch.Tensor target: The tensor from the TimeModule network.
        :param torch.Tensor prediction: The tensor from the FeatureModule network.
        :return torch.Tensor: KD MSE loss (embedding or target loss)
        """
        return F.mse_loss(input=prediction, target=target)

    def forward(self, phase_context: PhaseContext, network_outputs: NetworkOutputs) -> torch.Tensor:
        """
        Cascading forward pass that aggregates loss components based on the exact
        incremental phases defined in the original EINN training loop.

        :param PhaseContext phase_context: The context object indicating the current phase.
        :param NetworkOutputs network_outputs: Dataclass containing network predictions and derivatives.
        :return torch.Tensor: The weighted composite loss scalar.
        """
        total_loss = torch.tensor(data=0.0, dtype=torch.float32, device=phase_context.t.device)

        # Phase 1: Train Time Module (Solo Data, Aux & Monotonicity)
        if phase_context.phase_num >= 1:
            total_loss += self.weights['data_T'] * self.calc_data_loss(
                states=network_outputs.s_t, targets=phase_context.y
            )
            total_loss += self.weights['aux'] * self.calc_aux_loss(
                states=network_outputs.s_t, aux_targets=phase_context.aux_targets
            )
            total_loss += self.weights['mono'] * self.calc_monotonicity_loss(
                dS_dt=network_outputs.dS_dt_T_ode,
                inc_indices=self.mono_inc_indices,
                dec_indices=self.mono_dec_indices
            )

        # Phase 2: Add ODE Physics to Time Module
        if phase_context.phase_num >= 2:
            total_loss += self.weights['ode_T'] * self.calc_ode_loss(
                ds_dt_nn=network_outputs.ds_dt_T_nn, ds_dt_ode=network_outputs.ds_dt_T_ode
            )
            total_loss += self.weights['ode_future_T'] * self.calc_ode_loss(
                ds_dt_nn=network_outputs.ds_dt_future_T_nn, ds_dt_ode=network_outputs.ds_dt_future_T_ode
            )
            total_loss += self.weights['param'] * self.calc_parameter_smoothness_loss(
                params=network_outputs.params
            )

        # Phase 3: Add Feature Module (Data & Knowledge Distillation)
        if phase_context.phase_num >= 3:
            total_loss += self.weights['data_F'] * self.calc_data_loss(
                states=network_outputs.s_t_F, targets=phase_context.y
            )
            total_loss += self.weights['kd_target'] * self.calc_knowledge_distillation_loss(
                target=network_outputs.s_t.detach(), prediction=network_outputs.s_t_F
            )
            total_loss += self.weights['kd_emb'] * self.calc_knowledge_distillation_loss(
                target=network_outputs.e_t.detach(), prediction=network_outputs.e_t_F
            )

        # Phase 4: Final Fine-Tuning (Gradient Loss / Feature ODE)
        if phase_context.phase_num == 4:
            total_loss += self.weights['ode_F'] * self.calc_ode_loss(
                ds_dt_nn=network_outputs.ds_dt_F_nn, ds_dt_ode=network_outputs.ds_dt_F_ode
            )
            total_loss += self.weights['ode_future_F'] * self.calc_ode_loss(
                ds_dt_nn=network_outputs.ds_dt_future_F_nn, ds_dt_ode=network_outputs.ds_dt_future_F_ode
            )

        return total_loss
