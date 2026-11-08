import torch

from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.model.interface.phase_context import PhaseContext
from einn.model.interface.network_outputs import NetworkOutputs


class EINNForwardEngine:

    """
    Core execution engine for the EINN architecture.
    Responsible for routing data through the Time, Feature, and Output modules, computing empirical and
    analytical derivatives, and applying the gradient trick.
    """
    def __init__(self, train_config: EINNTrainConfig, scaling_stats: DataScalingStats):
        """
        Initializes the Forward Engine with the training configuration.

        :param EINNTrainConfig train_config: Configuration containing training hyperparameters (e.g., device).
        :param DataScalingStats scaling_stats: Data class containing runtime scaling statistics (means, stds, bounds).
        """
        self.config = train_config
        self.scaling_stats = scaling_stats

    @staticmethod
    def _get_jacobian_rows(outputs: torch.Tensor, inputs: torch.Tensor) -> list[torch.Tensor]:
        """
        Internal helper method to compute the gradients of each output feature w.r.t inputs.

        :param torch.Tensor outputs: The predicted tensor. Shape: [Batch, Seq_len, Features].
        :param torch.Tensor inputs: The input tensor (requires_grad=True).
        :return list[torch.Tensor]: A list of gradient tensors for each feature.
        """
        grads = []
        features_dim = outputs.shape[2]

        for i in range(features_dim):
            grad_i = torch.autograd.grad(
                outputs=outputs[:, :, i:i + 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(input=outputs[:, :, i:i + 1]),
                create_graph=True
            )[0]
            grads.append(grad_i)

        return grads

    def _compute_empirical_derivatives(self, outputs: torch.Tensor, time_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the time derivative (ds_t/dt or de_t/dt) using PyTorch autograd.

        :param torch.Tensor outputs: The predicted tensor. Shape: [Batch, Seq_len, Features].
        :param torch.Tensor time_tensor: The time input tensor. Shape: [Batch, Seq_len, 1].
        :return torch.Tensor: The computed derivative tensor. Shape: [Batch, Seq_len, Features].
        """
        grads = self._get_jacobian_rows(outputs=outputs, inputs=time_tensor)
        return torch.cat(tensors=grads, dim=-1)

    def _compute_feature_gradient_trick(
            self, s_t_f: torch.Tensor, e_t_f_grad: torch.Tensor, de_dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the chain-rule gradient trick for the Feature module to compute ds_t^F/dt.
        (We assume that e_t and e_t^F are very close to each other - we achieve this using L_emb loss function)
        Formula: ds_t^F/dt = (ds_t^F/de_t^F) * (de_t/dt)

        :param torch.Tensor s_t_f: The predicted states from the Feature module.
        :param torch.Tensor e_t_f_grad: The embeddings with requires_grad=True.
        :param torch.Tensor de_dt: The empirical derivative of embeddings w.r.t time from the Time module.
        :return torch.Tensor: The empirical time derivative of the Feature module's states.
        """
        ds_de_rows = self._get_jacobian_rows(outputs=s_t_f, inputs=e_t_f_grad)
        ds_dt_list = [(de_dt * ds_i_de).sum(dim=-1, keepdim=True) for ds_i_de in ds_de_rows]
        return torch.cat(tensors=ds_dt_list, dim=-1)

    def forward(self, context: PhaseContext) -> NetworkOutputs:
        """
        Executes the forward pass based on the current optimization phase.

        :param PhaseContext context: Contains the models, inputs, and phase identifier.
        :return NetworkOutputs: Dataclass containing all required tensors for the active phase's loss calculations.
        """
        out = NetworkOutputs()
        models = context.models

        x_input = context.x
        past_steps = x_input.shape[1]

        # Detach raw time to prevent autograd from traversing back to the data loader
        t_raw = context.t.clone().detach()

        # Extract scaling statistics generated during data preprocessing
        aux_mean = self.scaling_stats.aux_mean.to(t_raw.device)
        aux_std = self.scaling_stats.aux_std.to(t_raw.device)
        t_min = self.scaling_stats.t_min.to(t_raw.device)
        t_max = self.scaling_stats.t_max.to(t_raw.device)

        # Apply Min-Max scaling to time and enable gradient tracking
        t_range = (t_max - t_min + 1e-8)
        t_scaled = ((t_raw - t_min) / t_range).requires_grad_(True)

        # Time Module Processing
        e_t_full = models.time_module(t=t_scaled)
        s_t_full = models.output_module(e=e_t_full)

        # Compute derivative with respect to normalized time (ds/dt_norm)
        ds_t_dt_norm = self._compute_empirical_derivatives(outputs=s_t_full, time_tensor=t_scaled)

        # Scale derivative back to real physical time.
        # Since t_norm = t_raw / t_range, then dt_norm = dt_raw / t_range.
        # Therefore, ds/dt_raw = ds/dt_norm * (dt_norm/dt_raw) = ds/dt_norm / t_range.
        ds_t_dt_nn_full = ds_t_dt_norm / t_range

        # Inverse Z-score scaling to map neural states back to physical population fractions [0, 1]
        s_t_full_real = (s_t_full * aux_std) + aux_mean

        # Calculate analytical physical derivatives using the ODE model
        ode_solution_full = models.ode_model.get_derivatives(states=s_t_full_real, detach_params=False)

        # Scale analytical derivative to match the neural network's Z-scored loss space.
        # If real_state = (nn_state * std) + mean, then d(real_state)/dt = d(nn_state)/dt * std.
        # To compare ODE to NN in the loss function, we must divide the ODE output by std.
        ds_t_dt_ode_full = ode_solution_full.ds_dt / aux_std

        # Store past (historical) steps in the output container
        out.e_t = e_t_full[:, :past_steps, :]
        out.s_t = s_t_full[:, :past_steps, :]
        out.ds_dt_T_nn = ds_t_dt_nn_full[:, :past_steps, :]
        out.ds_dt_T_ode = ds_t_dt_ode_full[:, :past_steps, :]

        # Handle time-dependent (dynamic) vs static physical parameters
        if ode_solution_full.params.dim() == 3:
            out.params = ode_solution_full.params[:, :past_steps, :]
        else:
            out.params = ode_solution_full.params

        # Store future predictions if the curriculum has reached Phase 2
        if context.phase_num >= 2:
            out.ds_dt_future_T_nn = ds_t_dt_nn_full[:, past_steps:, :]
            out.ds_dt_future_T_ode = ds_t_dt_ode_full[:, past_steps:, :]

        # Feature Module Processing
        if context.phase_num >= 3:
            # The Feature module processes exogenous data. We detach time to prevent gradients
            # from flowing backwards into the Time Module through the time tensor.
            e_t_f_full = models.feature_module(x=x_input, t=t_scaled.detach(), mask=context.mask)

            # In Phase 4, the Feature Module's weights are completely frozen.
            # Detaching the embedding itself ensures no gradients flow back at all.
            if context.phase_num == 4:
                e_t_f_full = e_t_f_full.detach()

            s_t_f_full = models.output_module(e=e_t_f_full)

            out.e_t_F = e_t_f_full[:, :past_steps, :]
            out.s_t_F = s_t_f_full[:, :past_steps, :]

        # PHASE 4: Feature Gradient Trick and Final Physics Check
        if context.phase_num == 4:
            # Compute stable temporal derivative of the latent space using the Time Module
            de_dt_norm = self._compute_empirical_derivatives(outputs=e_t_full, time_tensor=t_scaled)
            de_dt_full = de_dt_norm / t_range

            # Prepare Feature Module embeddings for the chain rule by forcing gradient tracking
            e_t_f_grad_full = e_t_f_full.clone().detach().requires_grad_(True)
            s_t_f_grad_full = models.output_module(e=e_t_f_grad_full)

            # Apply the chain-rule trick to get smooth temporal derivatives for exogenous data
            ds_dt_f_nn_full = self._compute_feature_gradient_trick(
                s_t_f=s_t_f_grad_full, e_t_f_grad=e_t_f_grad_full, de_dt=de_dt_full
            )

            # Map back to physical domain and calculate ODE derivatives
            s_t_f_grad_full_real = (s_t_f_grad_full * aux_std) + aux_mean
            ds_dt_f_ode_full = models.ode_model.get_derivatives(states=s_t_f_grad_full_real,
                                                                detach_params=True).ds_dt / aux_std

            # Populate the final physics outputs for both past and future horizons
            out.ds_dt_F_nn = ds_dt_f_nn_full[:, :past_steps, :]
            out.ds_dt_F_ode = ds_dt_f_ode_full[:, :past_steps, :]
            out.ds_dt_future_F_nn = ds_dt_f_nn_full[:, past_steps:, :]
            out.ds_dt_future_F_ode = ds_dt_f_ode_full[:, past_steps:, :]

        return out
