import torch

from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext


class EINNForwardEngine:
    """
    Phase-aware execution engine for single-model EINN forward passes and derivatives.

    Computes normalized compartment predictions and latent embeddings for the Time and
    Feature branches, scales autograd and mechanistic ODE derivatives back to physical
    time/state units, and applies the chain-rule gradient trick on Feature embeddings
    in Phase 4.
    """

    def __init__(self, train_config: EINNTrainConfig, scaling_stats: DataScalingStats) -> None:
        """
        Initialize the forward engine with training hyperparameters and scaling statistics.

        :param EINNTrainConfig train_config: Training configuration providing `future_steps`
            and execution settings.
        :param DataScalingStats scaling_stats: Training-fitted temporal bounds (`t_min`, `t_max`)
            and auxiliary compartment statistics (`aux_mean`, `aux_std`) for unit conversion.
        """
        self.config: EINNTrainConfig = train_config
        self.scaling_stats: DataScalingStats = scaling_stats

    @staticmethod
    def _get_jacobian_rows(outputs: torch.Tensor, inputs: torch.Tensor) -> list[torch.Tensor]:
        """
        Compute gradients `d(outputs[..., i])/d(inputs)` for each output channel `i`.

        Retains higher-order graph connectivity (`create_graph=True`) so computed derivatives
        remain differentiable with respect to network parameters and input tensors.

        :param torch.Tensor outputs: Output tensor of shape `(B, T, Features)`.
        :param torch.Tensor inputs: Differentiable input tensor with `requires_grad=True`.
        :return list[torch.Tensor]: List of gradient tensors matching `inputs.shape`, one per
            last-dimension feature channel.
        """
        grads: list[torch.Tensor] = []
        features_dim = outputs.shape[2]

        for i in range(features_dim):
            grad_i = torch.autograd.grad(
                outputs=outputs[:, :, i:i + 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(input=outputs[:, :, i:i + 1]),
                create_graph=True,
            )[0]
            grads.append(grad_i)

        return grads

    def _compute_empirical_derivatives(
        self, outputs: torch.Tensor, time_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute exact autograd time derivatives `d(outputs)/d(time_tensor)` across all channels.

        :param torch.Tensor outputs: Differentiable tensor of shape `(B, T_total, Features)`.
        :param torch.Tensor time_tensor: Normalized time tensor of shape `(B, T_total, 1)`
            with `requires_grad=True`.
        :return torch.Tensor: Derivative tensor of shape `(B, T_total, Features)` with respect
            to `time_tensor`.
        """
        grads = self._get_jacobian_rows(outputs=outputs, inputs=time_tensor)
        return torch.cat(tensors=grads, dim=-1)

    def _compute_feature_gradient_trick(
        self,
        s_t_f: torch.Tensor,
        e_t_f_grad: torch.Tensor,
        de_dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate `ds_t^F / dt` via the chain-rule gradient trick: `J_out(e_t^F) * (de_t^T / dt)`.

        Relies on embedding distillation (`L_emb`) keeping Feature embeddings `e_t^F` aligned
        with Time embeddings `e_t^T`, projecting `de_dt` through each row of the OutputModule
        Jacobian `d(s_t^F) / d(e_t^F)`.

        :param torch.Tensor s_t_f: Feature-branch predicted states of shape `(B, T_total, d_s)`.
        :param torch.Tensor e_t_f_grad: Differentiable Feature embeddings of shape `(B, T_total, d_e)`
            with `requires_grad=True`.
        :param torch.Tensor de_dt: Time-branch embedding time derivative in physical time units,
            of shape `(B, T_total, d_e)`.
        :return torch.Tensor: Approximated Feature state time derivative of shape `(B, T_total, d_s)`.
        """
        ds_de_rows = self._get_jacobian_rows(outputs=s_t_f, inputs=e_t_f_grad)
        ds_dt_list = [(de_dt * ds_i_de).sum(dim=-1, keepdim=True) for ds_i_de in ds_de_rows]
        return torch.cat(tensors=ds_dt_list, dim=-1)

    def forward(self, context: PhaseContext) -> NetworkOutputs:
        """
        Execute the phase-dependent single-model EINN forward pass and return outputs/derivatives.

        Extends the time grid by `config.future_steps` in Phases 2-4 for future ODE residual
        supervision, evaluates Time-branch states and derivatives in Phases 1-4, evaluates
        Feature-branch embeddings and states in Phases 3-4, and computes Feature-branch
        gradient-trick ODE residuals in Phase 4.

        :param PhaseContext context: Current training step context containing `phase_num`,
            input/target/time tensors, mask, and model container.
        :return NetworkOutputs: Populated forward-pass outputs for the active training phase.
        """
        out = NetworkOutputs()
        models = context.models

        x_input = context.x
        past_steps = x_input.shape[1]  # Full sequence length on the training grid (e.g., 200).
        t_raw = context.t.clone().detach()
        if t_raw.ndim != 3 or t_raw.shape != (x_input.shape[0], past_steps, 1):
            raise ValueError(
                "Expected context.t shape [batch, sequence_length, 1] aligned with X, "
                f"got X={tuple(x_input.shape)} and t={tuple(t_raw.shape)}."
            )

        # Append `future_steps` new time points past the full grid in Phases 2-4 so future ODE
        # losses evaluate on unseen extrapolation steps rather than an empty slice.
        if context.phase_num >= 2:
            future_steps = int(self.config.future_steps)
            if future_steps <= 0:
                raise ValueError("future_steps must be positive in phases 2–4.")
            if past_steps < 2:
                raise ValueError("At least two time points are required.")

            step = t_raw[:, -1:, :] - t_raw[:, -2:-1, :]
            if not bool(torch.isfinite(step).all() and (step > 0).all()):
                raise ValueError("The last raw time step must be finite and positive.")
            increments = torch.arange(
                1, future_steps + 1, device=t_raw.device, dtype=t_raw.dtype
            ).view(1, -1, 1)
            t_raw = torch.cat((t_raw, t_raw[:, -1:, :] + step * increments), dim=1)

        # Extract training-fitted scaling parameters onto the active device.
        aux_mean = self.scaling_stats.aux_mean.to(t_raw.device)
        aux_std = self.scaling_stats.aux_std.to(t_raw.device)
        t_min = self.scaling_stats.t_min.to(t_raw.device)
        t_max = self.scaling_stats.t_max.to(t_raw.device)

        # Min-max scale time coordinates and enable gradient tracking in normalized time space.
        t_range = (t_max - t_min).clamp_min(1e-8)
        t_scaled = ((t_raw - t_min) / t_range).requires_grad_(True)

        # Phases 1-4: TimeModule -> OutputModule forward pass.
        e_t_full = models.time_module(t=t_scaled)
        s_t_full = models.output_module(e=e_t_full)

        # Differentiate w.r.t. normalized time, then apply the chain rule to obtain physical ds/dt.
        ds_t_dt_norm = self._compute_empirical_derivatives(outputs=s_t_full, time_tensor=t_scaled)
        ds_t_dt_nn_full = ds_t_dt_norm / t_range

        # Convert normalized states to physical units, evaluate ODE RHS, and normalize derivatives.
        s_t_full_real = (s_t_full * aux_std) + aux_mean
        ode_solution_full = models.ode_model.get_derivatives(states=s_t_full_real, detach_params=False)
        ds_t_dt_ode_full = ode_solution_full.ds_dt / aux_std

        # Slice past_steps back to the original training horizon for NetworkOutputs.
        out.e_t = e_t_full[:, :past_steps, :]
        out.s_t = s_t_full[:, :past_steps, :]
        out.ds_dt_T_nn = ds_t_dt_nn_full[:, :past_steps, :]
        out.ds_dt_T_ode = ds_t_dt_ode_full[:, :past_steps, :]

        if ode_solution_full.params.dim() == 3:
            out.params = ode_solution_full.params[:, :past_steps, :]
        else:
            out.params = ode_solution_full.params

        if context.phase_num >= 2:
            out.ds_dt_future_T_nn = ds_t_dt_nn_full[:, past_steps:, :]
            out.ds_dt_future_T_ode = ds_t_dt_ode_full[:, past_steps:, :]
            if out.ds_dt_future_T_nn.shape[1] != future_steps:
                raise RuntimeError("Time future ODE block has an incorrect length.")

        if context.phase_num >= 3:
            # Detach t_scaled so FeatureModule decoding does not backpropagate through time coordinates.
            e_t_f_full = models.feature_module(x=x_input, t=t_scaled.detach(), mask=context.mask)

            if context.phase_num == 4:
                # Freeze FeatureModule embeddings during Phase 4 fine-tuning of ODE/Output modules.
                e_t_f_full = e_t_f_full.detach()

            s_t_f_full = models.output_module(e=e_t_f_full)

            out.e_t_F = e_t_f_full[:, :past_steps, :]
            out.s_t_F = s_t_f_full[:, :past_steps, :]

        if context.phase_num == 4:
            # Rescale latent embedding derivative de/dt to physical time units for the gradient trick.
            de_dt_norm = self._compute_empirical_derivatives(outputs=e_t_full, time_tensor=t_scaled)
            de_dt_full = de_dt_norm / t_range

            e_t_f_grad_full = e_t_f_full.clone().detach().requires_grad_(True)
            s_t_f_grad_full = models.output_module(e=e_t_f_grad_full)

            # Evaluate Feature-branch state derivatives across the full (past + future) time grid.
            ds_dt_f_nn_full = self._compute_feature_gradient_trick(
                s_t_f=s_t_f_grad_full, e_t_f_grad=e_t_f_grad_full, de_dt=de_dt_full
            )

            s_t_f_grad_full_real = (s_t_f_grad_full * aux_std) + aux_mean
            ds_dt_f_ode_full = models.ode_model.get_derivatives(
                states=s_t_f_grad_full_real,
                detach_params=True,
            ).ds_dt / aux_std

            out.ds_dt_F_nn = ds_dt_f_nn_full[:, :past_steps, :]
            out.ds_dt_F_ode = ds_dt_f_ode_full[:, :past_steps, :]
            out.ds_dt_future_F_nn = ds_dt_f_nn_full[:, past_steps:, :]
            out.ds_dt_future_F_ode = ds_dt_f_ode_full[:, past_steps:, :]
            if out.ds_dt_future_F_nn.shape[1] != future_steps:
                raise RuntimeError("Feature future ODE block has an incorrect length.")

        return out
