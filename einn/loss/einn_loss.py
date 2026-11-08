import torch
import torch.nn as nn
import torch.nn.functional as F

from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext
from einn.ode.base_ode_model import BaseODEModel


class EINNLoss(nn.Module):
    """
    Computes the composite loss for the EINN architecture.
    """

    def __init__(self, config: EINNTrainConfig, ode_model: BaseODEModel, scaling_stats: DataScalingStats):
        """
        Initializes the EINNLoss module.

        :param EINNTrainConfig config: The configuration object containing loss weights.
        :param BaseODEModel ode_model: The epidemiological ODE model providing monotonicity rules.
        :param DataScalingStats scaling_stats: The runtime scaling statistics, used to retrieve initial states.
        """
        super().__init__()
        self.config = config
        self.ode_model = ode_model
        self.weights = config.loss_weights
        self.scaling_stats = scaling_stats

        self.mono_dec_indices = self.ode_model.monotonicity_indices["decreasing"]
        self.mono_inc_indices = self.ode_model.monotonicity_indices["increasing"]

    def calc_data_loss(self, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Mean Squared Error (MSE) between predicted states and observed data.
        Assumes targets map to the first dim(targets) compartments (e.g., Infected).

        :param torch.Tensor states: Predicted compartment states. Shape: [Batch, Seq_len, d_s].
        :param torch.Tensor targets: Ground truth observations. Shape: [Batch, Seq_len, target_dim].
        :return torch.Tensor: Data MSE loss.
        """
        if states.numel() == 0 or targets.numel() == 0:
            return torch.tensor(data=0.0, dtype=torch.float32, device=states.device)

        idx = self.ode_model.target_idx

        # Slice the states tensor to extract only the target compartment's predictions.
        # :idx:idx+1 ensures the slice retains its trailing dimension (e.g., [Batch, Seq_len, 1]).
        return F.mse_loss(input=states[:, :, :idx:idx + 1], target=targets)

    @staticmethod
    def calc_aux_loss(states: torch.Tensor, aux_targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the auxiliary loss against ideal ODE trajectories for Phase 1.

        :param torch.Tensor states: Predicted compartment states. Shape: [Batch, Seq_len, d_s].
        :param torch.Tensor aux_targets: Ideal simulated compartment states.
        :return torch.Tensor: Auxiliary MSE loss.
        """
        if states.numel() == 0:
            return torch.tensor(data=0.0, dtype=torch.float32, device=states.device)

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
        if ds_dt_nn.numel() == 0:
            return torch.tensor(data=0.0, dtype=torch.float32, device=ds_dt_nn.device)
        return F.mse_loss(input=ds_dt_nn, target=ds_dt_ode)

    @staticmethod
    def calc_monotonicity_loss(
            ds_dt: torch.Tensor, inc_indices: list[int], dec_indices: list[int]
    ) -> torch.Tensor:
        """
        Applies a squared asymmetric ReLU penalty to enforce monotonic compartments.
        Penalizes when decreasing states (e.g., S) have positive derivatives, and when
        increasing states (e.g., R, M) have negative derivatives.

        :param torch.Tensor ds_dt: Time derivative of all the states. Shape: [Batch, Seq_len, d_s].
        :param list[int] inc_indices: List of indices that must increase.
        :param list[int] dec_indices: List of indices that must decrease.
        :return torch.Tensor: A scalar tensor representing the monotonicity penalty.
        """
        if ds_dt.numel() == 0:
            return torch.tensor(data=0.0, dtype=torch.float32, device=ds_dt.device)

        penalty = torch.tensor(data=0.0, dtype=torch.float32, device=ds_dt.device)

        # Increasing compartments: apply penalty if derivative is negative
        for idx in inc_indices:
            val = -ds_dt[:, :, idx]
            penalty += torch.mean(input=(val * torch.relu(input=val)) ** 2)

        # Decreasing compartments: apply penalty if derivative is positive
        for idx in dec_indices:
            val = ds_dt[:, :, idx]
            penalty += torch.mean(input=(val * torch.relu(input=val)) ** 2)

        return penalty

    @staticmethod
    def calc_parameter_smoothness_loss(params: torch.Tensor) -> torch.Tensor:
        """
        Penalizes large jumps in time-dependent physical parameters between consecutive steps.

        :param torch.Tensor params: Time-dependent parameter tensor. Shape: [Batch, Seq_len, d_p].
        :return torch.Tensor: Parameter smoothness loss.
        """
        # For time-independent parameters
        if params.dim() < 2 or params.shape[1] <= 1:
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
        if prediction.numel() == 0:
            return torch.tensor(data=0.0, dtype=torch.float32, device=prediction.device)

        return F.mse_loss(input=prediction, target=target)

    @staticmethod
    def calc_ic_loss(states: torch.Tensor, t: torch.Tensor, initial_states: list[float]) -> torch.Tensor:
        """
        Calculates the Initial Condition (IC) loss to enforce the physical starting state of the system at t=0.
        This anchors the neural network's predictions at the beginning of the trajectory to the known biological baseline.

        :param torch.Tensor states: The predicted physical compartments over time. Shape: [Batch, Seq_len, d_s].
        :param torch.Tensor t: The time steps associated with the predictions. Shape: [Batch, Seq_len, 1].
        :param list[float] initial_states: A list of the normalized ground truth initial compartment values. Length: d_s.
        :return torch.Tensor: The MSE loss between the predicted states at t = 0 and the true initial_states.
        """
        if states.numel() == 0 or t.numel() == 0 or not initial_states:
            return torch.tensor(0.0, device=states.device)

        # Create a boolean mask to isolate the exact time steps that represent t = 0
        # Using a small tolerance (1e-3) is safer than strict equality for floating-point tensors
        # Squeeze removes the trailing dimension of 't' for boolean indexing
        mask_t0 = (t < 1e-3).squeeze(-1)

        # If the batch does not contain t = 0 (e.g., in a sliding window far in the future), IC loss is zero
        if not mask_t0.any():
            return torch.tensor(0.0, device=states.device)

        # Extract only the predicted state vectors that correspond to t = 0
        states_at_t0 = states[mask_t0]

        # Convert the list of ground truth initial states to a tensor.
        # Expand it to match the batch dimension of 'states_at_t0' so they can be compared directly.
        target_ic = torch.tensor(initial_states, dtype=torch.float32, device=states.device) \
            .unsqueeze(0) \
            .expand(states_at_t0.shape[0], -1)

        return F.mse_loss(states_at_t0, target_ic)

    def forward(self, phase_context: PhaseContext,
                network_outputs: NetworkOutputs) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Cascading forward pass that aggregates loss components based on the exact
        incremental phases defined in the original EINN training loop.

        :param PhaseContext phase_context: The context object indicating the current phase.
        :param NetworkOutputs network_outputs: Dataclass containing network predictions and derivatives.
        :return tuple[torch.Tensor, dict[str, float]]: The weighted composite loss scalar and a dictionary
            containing the loss components
        """
        total_loss = torch.tensor(data=0.0, dtype=torch.float32, device=phase_context.t.device)
        loss_dict = {}

        def add_loss(component_name: str, loss_tensor: torch.Tensor):
            nonlocal total_loss
            weight = self.weights.get(component_name, 0.0)
            if weight > 0:
                weighted_loss = weight * loss_tensor
                total_loss += weighted_loss
                loss_dict[component_name] = weighted_loss.item()

        # Phase 1: Train Time Module (Solo Data, Aux & Monotonicity)
        if phase_context.phase_num >= 1:
            add_loss(component_name='data_T',
                     loss_tensor=self.calc_data_loss(
                         states=network_outputs.s_t,
                         targets=phase_context.y))
            add_loss(component_name='aux',
                     loss_tensor=self.calc_aux_loss(
                         states=network_outputs.s_t,
                         aux_targets=phase_context.aux_targets))
            add_loss(component_name='mono',
                     loss_tensor=self.calc_monotonicity_loss(
                         ds_dt=network_outputs.ds_dt_T_ode,
                         inc_indices=self.mono_inc_indices,
                         dec_indices=self.mono_dec_indices))
            add_loss(component_name='ic',
                     loss_tensor=self.calc_ic_loss(
                         states=network_outputs.s_t,
                         t=phase_context.t,
                         initial_states=self.scaling_stats.initial_states))

        # Phase 2: Add ODE Physics to Time Module
        if phase_context.phase_num >= 2:
            add_loss(component_name='ode_T',
                     loss_tensor=self.calc_ode_loss(
                         ds_dt_nn=network_outputs.ds_dt_T_nn,
                         ds_dt_ode=network_outputs.ds_dt_T_ode))
            add_loss(component_name='ode_future_T',
                     loss_tensor=self.calc_ode_loss(
                         ds_dt_nn=network_outputs.ds_dt_future_T_nn,
                         ds_dt_ode=network_outputs.ds_dt_future_T_ode))
            add_loss(component_name='param',
                     loss_tensor=self.calc_parameter_smoothness_loss(
                         params=network_outputs.params))

        # Phase 3: Add Feature Module (Data & Knowledge Distillation)
        if phase_context.phase_num >= 3:
            add_loss(component_name='data_F',
                     loss_tensor=self.calc_data_loss(
                         states=network_outputs.s_t_F, targets=phase_context.y))
            add_loss(component_name='kd_target',
                     loss_tensor=self.calc_knowledge_distillation_loss(
                         target=network_outputs.s_t.detach(),
                         prediction=network_outputs.s_t_F))
            add_loss(component_name='kd_emb',
                     loss_tensor=self.calc_knowledge_distillation_loss(
                         target=network_outputs.e_t.detach(),
                         prediction=network_outputs.e_t_F))
            add_loss(component_name='ic_F',
                     loss_tensor=self.calc_ic_loss(
                         states=network_outputs.s_t_F,
                         t=phase_context.t,
                         initial_states=self.scaling_stats.initial_states))

        # Phase 4: Final Fine-Tuning (Gradient Loss / Feature ODE)
        if phase_context.phase_num == 4:
            add_loss(component_name='ode_F',
                     loss_tensor=self.calc_ode_loss(
                         ds_dt_nn=network_outputs.ds_dt_F_nn,
                         ds_dt_ode=network_outputs.ds_dt_F_ode))
            add_loss(component_name='ode_future_F',
                     loss_tensor=self.calc_ode_loss(
                         ds_dt_nn=network_outputs.ds_dt_future_F_nn,
                         ds_dt_ode=network_outputs.ds_dt_future_F_ode))

        return total_loss, loss_dict
