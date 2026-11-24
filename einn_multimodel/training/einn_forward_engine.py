"""
Forward pass and derivative computation engine for multi-model EINN training.

Routes batches by epidemiological model label to model-specific TimeModules and ODE
models, enforces structural dummy compartments in normalized state space, extends the
time grid into the extrapolation horizon for Phases 2-4, computes autograd Time-branch
derivatives, and evaluates the Feature-branch gradient trick in Phase 4 (thesis
Sections 2.2 and 2.4).
"""
import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.model.interface.einn_models import EINNModels
from einn_multimodel.model.interface.network_outputs import NetworkOutputs
from einn_multimodel.model.interface.phase_context import PhaseContext


class EINNForwardEngine:
    """
    Phase-aware forward engine for multi-model EINN training.

    Computes normalized states and latent embeddings for the Time and Feature branches,
    converts derivatives between normalized and physical units via training-fitted scaling
    statistics, and applies the chain-rule gradient trick on Feature embeddings in Phase 4
    (thesis Section 2.2).
    """

    def __init__(self, train_config: EINNTrainConfig, scaling_stats: DataScalingStats) -> None:
        """
        Store training hyperparameters and training-fitted scaling statistics.

        :param EINNTrainConfig train_config: Training configuration providing future_steps and device.
        :param DataScalingStats scaling_stats: Training-fitted temporal bounds (t_min, t_max)
            and compartment statistics (aux_mean, aux_std) for unit conversion.
        """
        self.config: EINNTrainConfig = train_config
        self.scaling_stats: DataScalingStats = scaling_stats

    @staticmethod
    def _get_jacobian_rows(outputs: torch.Tensor, inputs: torch.Tensor) -> list[torch.Tensor]:
        """
        Compute row-wise gradients d(outputs[..., k])/d(inputs) for each last-axis channel k.

        Retains higher-order graph connectivity (create_graph=True, retain_graph=True) so
        derivatives remain differentiable with respect to network parameters and inputs.

        :param torch.Tensor outputs: Output tensor of shape (..., C) to differentiate.
        :param torch.Tensor inputs: Differentiable input tensor of shape (..., D) with requires_grad=True.
        :return list[torch.Tensor]: List of C gradient tensors, each matching inputs.shape.
        """
        gradients = []
        for feature_idx in range(outputs.shape[-1]):
            feature_output = outputs[..., feature_idx:feature_idx + 1]
            if not feature_output.requires_grad:
                gradients.append(torch.zeros_like(inputs))
                continue
            grad = torch.autograd.grad(
                outputs=feature_output,
                inputs=inputs,
                grad_outputs=torch.ones_like(feature_output),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            gradients.append(torch.zeros_like(inputs) if grad is None else grad)
        return gradients

    def _compute_empirical_derivatives(
        self, outputs: torch.Tensor, time_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute exact autograd derivatives d(outputs)/d(time_tensor) across all output channels.

        :param torch.Tensor outputs: Differentiable tensor of shape (B, T_total, C).
        :param torch.Tensor time_tensor: Normalized time tensor of shape (B, T_total, 1)
            with requires_grad=True.
        :return torch.Tensor: Derivative tensor of shape (B, T_total, C) with respect to time_tensor.
        """
        return torch.cat(self._get_jacobian_rows(outputs, time_tensor), dim=-1)

    def _compute_feature_gradient_trick(
        self,
        states: torch.Tensor,
        embedding: torch.Tensor,
        de_dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate d(s_F)/dt via the EINN gradient trick: J_out(e_F) * d(e_T)/dt.

        Multiplies each row of the OutputModule Jacobian J_out(e_t^F) = d(s_t^F)/d(e_t^F) by the
        Time-branch latent derivative d(e_t^T)/dt along the embedding axis d_e (thesis Section 2.2).

        :param torch.Tensor states: Feature-branch output states s_t^F of shape (B, T_total, d_s).
        :param torch.Tensor embedding: Differentiable Feature embeddings e_t^F of shape (B, T_total, d_e).
        :param torch.Tensor de_dt: Time-branch embedding derivative d(e_t^T)/dt in physical time
            units, of shape (B, T_total, d_e).
        :return torch.Tensor: Projected Feature state derivatives of shape (B, T_total, d_s).
        """
        ds_de_rows = self._get_jacobian_rows(states, embedding)
        return torch.cat(
            [(de_dt * row).sum(dim=-1, keepdim=True) for row in ds_de_rows], dim=-1
        )

    @staticmethod
    def _labels(context: PhaseContext) -> torch.Tensor:
        """
        Extract and validate per-sample integer model labels of shape (B,) from PhaseContext.

        :param PhaseContext context: Current step context containing true_model_label and models.
        :return torch.Tensor: Validated int64 model ID tensor of shape (B,).
        :raises ValueError: If true_model_label is missing for a multi-model run or contains
            unknown model IDs outside context.models.model_to_id.
        """
        labels = context.true_model_label
        if labels is None:
            if len(context.models.model_to_id) != 1:
                raise ValueError("true_model_label is required for multiple models.")
            labels = torch.zeros(
                context.x.size(0), dtype=torch.long, device=context.x.device
            )
        labels = labels.reshape(-1).long()
        allowed_ids = set(context.models.model_to_id.values())
        observed_ids = set(labels.detach().cpu().tolist())
        invalid_ids = observed_ids.difference(allowed_ids)
        if invalid_ids:
            raise ValueError(
                f"Batch contains unknown true-model ids: {sorted(invalid_ids)}"
            )
        return labels

    def _route_time_modules(
        self, models: EINNModels, labels: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Route each batch sample to its corresponding model-specific TimeModule.

        :param EINNModels models: Multi-model EINN container holding time_modules and model_to_id.
        :param torch.Tensor labels: Integer model IDs of shape (B,).
        :param torch.Tensor t: Normalized time grid of shape (B, T_total, 1).
        :return torch.Tensor: Routed Time-branch embeddings e_t^T of shape (B, T_total, d_e).
        """
        embedding_dim = models.feature_module.decoder.dim_out
        result = t.new_zeros((t.size(0), t.size(1), embedding_dim))
        for name, module in models.time_modules.items():
            model_id = models.model_to_id[name]
            indices = torch.where(labels == model_id)[0]
            if indices.numel() == 0:
                continue
            subset = module(t=t.index_select(0, indices))
            result = result.index_copy(0, indices, subset)
        return result

    def _route_ode_models(
        self,
        models: EINNModels,
        labels: torch.Tensor,
        states_real: torch.Tensor,
        detach_params: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Route physical-domain compartment states to each sample's assigned ODE model RHS.

        :param EINNModels models: Multi-model EINN container holding ode_models and model_to_id.
        :param torch.Tensor labels: Integer model IDs of shape (B,).
        :param torch.Tensor states_real: Compartment states in physical units of shape (B, T_total, d_s).
        :param bool detach_params: If True, detaches ODE parameters before RHS evaluation
            (used in Phase 4 when fine-tuning ODE parameters/OutputModule against Feature derivatives).
        :return tuple[torch.Tensor, dict[str, torch.Tensor]]: Pair (ds_dt_ode_real, params_by_model)
            with physical RHS derivatives of shape (B, T_total, d_s) and per-model parameter tensors.
        """
        derivative_result = torch.zeros_like(states_real)
        params_by_model: dict[str, torch.Tensor] = {}
        for name, model in models.ode_models.items():
            model_id = models.model_to_id[name]
            indices = torch.where(labels == model_id)[0]
            if indices.numel() == 0:
                continue
            solution = model.get_derivatives(
                states=states_real.index_select(0, indices),
                detach_params=detach_params,
            )
            derivative_result = derivative_result.index_copy(0, indices, solution.ds_dt)
            params_by_model[name] = solution.params
        return derivative_result, params_by_model

    @staticmethod
    def _enforce_dummy_states(
        states_normalized: torch.Tensor,
        labels: torch.Tensor,
        models: EINNModels,
        aux_mean: torch.Tensor,
        aux_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Set structural dummy compartments to their standardized physical zero value in normalized space.

        Converts physical 0.0 via (0.0 - aux_mean[idx]) / aux_std[idx] so padded compartments
        (such as E in SIR-SEIR or the third channel in SIR-SIS) remain constant across time
        (thesis Section 2.4).

        :param torch.Tensor states_normalized: Normalized compartment states of shape (B, T_total, d_s).
        :param torch.Tensor labels: Integer model IDs of shape (B,).
        :param EINNModels models: Multi-model EINN container exposing ode_models and model_to_id.
        :param torch.Tensor aux_mean: Z-score means of shape (d_s,).
        :param torch.Tensor aux_std: Z-score standard deviations of shape (d_s,).
        :return torch.Tensor: Cloned normalized state tensor of shape (B, T_total, d_s) with
            dummy compartments fixed.
        """
        states = states_normalized.clone()
        for name, model in models.ode_models.items():
            dummy_indices = getattr(model, "dummy_state_indices", ())
            if not dummy_indices:
                continue
            indices = torch.where(labels == models.model_to_id[name])[0]
            if indices.numel() == 0:
                continue
            for state_idx in dummy_indices:
                normalized_zero = (0.0 - aux_mean[state_idx]) / aux_std[state_idx]
                states[indices, :, state_idx] = normalized_zero
        return states

    def forward(self, context: PhaseContext) -> NetworkOutputs:
        """
        Execute the phase-dependent EINN forward pass and return states, embeddings, and derivatives.

        Extends the time horizon by config.future_steps in Phases 2-4 to evaluate future ODE
        residuals, computes Time-branch autograd derivatives in Phases 2-4, runs FeatureModule
        and Feature-branch decoding in Phases 3-4, and computes the Feature gradient-trick ODE
        residuals in Phase 4 (thesis Sections 2.2 and 2.4).

        :param PhaseContext context: Current step context with phase_num, batch tensors, and labels.
        :return NetworkOutputs: Populated forward-pass outputs for the active training phase.
        :raises ValueError: If context.t shape does not match (B, full_steps, 1), or if
            future_steps / time_step assumptions are invalid in Phases 2-4.
        :raises RuntimeError: If future ODE derivative tensors do not have length future_steps.
        """
        out = NetworkOutputs()
        models = context.models
        labels = self._labels(context)
        x_input = context.x
        full_steps = x_input.shape[1]

        t_raw = context.t.detach().clone()
        if t_raw.ndim != 3 or t_raw.shape != (x_input.shape[0], full_steps, 1):
            raise ValueError(
                "Expected context.t shape [batch, sequence_length, 1] aligned with X, "
                f"got X={tuple(x_input.shape)} and t={tuple(t_raw.shape)}."
            )

        # Extend time coordinates past full_steps in Phases 2-4 so extrapolation ODE losses
        # operate on unseen future timesteps rather than an empty slice.
        if context.phase_num >= 2:
            future_steps = int(self.config.future_steps)
            if future_steps <= 0:
                raise ValueError(
                    "future_steps must be positive in phases 2–4 when "
                    "future ODE losses are enabled."
                )
            if full_steps < 2:
                raise ValueError("At least two time points are needed to infer the time step.")
            time_step = t_raw[:, -1:, :] - t_raw[:, -2:-1, :]
            if not bool(torch.isfinite(time_step).all() and (time_step > 0).all()):
                raise ValueError("The last raw time step must be finite and positive.")
            increments = torch.arange(
                1, future_steps + 1, device=t_raw.device, dtype=t_raw.dtype
            ).view(1, -1, 1)
            future_t_raw = t_raw[:, -1:, :] + time_step * increments
            t_raw = torch.cat((t_raw, future_t_raw), dim=1)  # Shape: (B, full_steps + future_steps, 1)

        # Normalize time coordinates into [0, 1] and enable autograd for Time-branch derivatives.
        aux_mean = self.scaling_stats.aux_mean.to(t_raw.device)
        aux_std = self.scaling_stats.aux_std.to(t_raw.device)
        t_min = self.scaling_stats.t_min.to(t_raw.device)
        t_max = self.scaling_stats.t_max.to(t_raw.device)
        t_range = (t_max - t_min).clamp_min(1e-8)
        t_scaled = ((t_raw - t_min) / t_range).requires_grad_(True)

        # Route each sample to its model-specific TimeModule -> e_t^T (B, T_total, d_e) -> s_t^T (B, T_total, d_s).
        e_t_full = self._route_time_modules(models, labels, t_scaled)
        s_t_full = models.output_module(e=e_t_full)
        s_t_full = self._enforce_dummy_states(
            s_t_full, labels, models, aux_mean, aux_std
        )

        # Convert autograd derivative d(s_t^T)/d(t_scaled) to physical time units and compare with ODE RHS.
        ds_dt_t_norm = self._compute_empirical_derivatives(s_t_full, t_scaled)
        ds_dt_t_nn = ds_dt_t_norm / t_range
        s_t_real = s_t_full * aux_std + aux_mean
        ds_dt_t_ode_real, params_by_model = self._route_ode_models(
            models, labels, s_t_real, detach_params=False
        )
        ds_dt_t_ode = ds_dt_t_ode_real / aux_std

        out.e_t = e_t_full[:, :full_steps]
        out.s_t = s_t_full[:, :full_steps]
        out.ds_dt_T_nn = ds_dt_t_nn[:, :full_steps]
        out.ds_dt_T_ode = ds_dt_t_ode[:, :full_steps]
        out.params = params_by_model
        if context.phase_num >= 2:
            out.ds_dt_future_T_nn = ds_dt_t_nn[:, full_steps:]
            out.ds_dt_future_T_ode = ds_dt_t_ode[:, full_steps:]
            if out.ds_dt_future_T_nn.shape[1] != future_steps:
                raise RuntimeError(
                    "Time future ODE block has an incorrect sequence length: "
                    f"{out.ds_dt_future_T_nn.shape[1]} != {future_steps}."
                )

        # Phases 3-4: Run shared FeatureModule on historical x and query times t_scaled.
        if context.phase_num >= 3:
            feature_outputs = models.feature_module(
                x=x_input,
                t=t_scaled.detach(),
                mask=context.mask,
                true_model_label=labels,
            )
            e_t_f_full = feature_outputs.embeddings
            if context.phase_num == 4:
                # Freeze FeatureModule embeddings during Phase 4 ODE/OutputModule fine-tuning.
                e_t_f_full = e_t_f_full.detach()
            s_t_f_full = models.output_module(e=e_t_f_full)
            s_t_f_full = self._enforce_dummy_states(
                s_t_f_full, labels, models, aux_mean, aux_std
            )

            out.e_t_F = e_t_f_full[:, :full_steps]
            out.s_t_F = s_t_f_full[:, :full_steps]

        # Phase 4: Evaluate Feature-branch ODE residuals using the EINN gradient trick (Section 2.2).
        if context.phase_num == 4:
            de_dt_norm = self._compute_empirical_derivatives(e_t_full, t_scaled)
            de_dt = de_dt_norm / t_range
            e_t_f_grad = e_t_f_full.clone().detach().requires_grad_(True)
            s_t_f_grad = models.output_module(e=e_t_f_grad)
            s_t_f_grad = self._enforce_dummy_states(
                s_t_f_grad, labels, models, aux_mean, aux_std
            )
            ds_dt_f_nn = self._compute_feature_gradient_trick(
                states=s_t_f_grad,
                embedding=e_t_f_grad,
                de_dt=de_dt,
            )
            s_t_f_real = s_t_f_grad * aux_std + aux_mean
            ds_dt_f_ode_real, _ = self._route_ode_models(
                models, labels, s_t_f_real, detach_params=True
            )
            ds_dt_f_ode = ds_dt_f_ode_real / aux_std

            out.ds_dt_F_nn = ds_dt_f_nn[:, :full_steps]
            out.ds_dt_F_ode = ds_dt_f_ode[:, :full_steps]
            out.ds_dt_future_F_nn = ds_dt_f_nn[:, full_steps:]
            out.ds_dt_future_F_ode = ds_dt_f_ode[:, full_steps:]
            if out.ds_dt_future_F_nn.shape[1] != future_steps:
                raise RuntimeError(
                    "Feature future ODE block has an incorrect sequence length: "
                    f"{out.ds_dt_future_F_nn.shape[1]} != {future_steps}."
                )

        return out
