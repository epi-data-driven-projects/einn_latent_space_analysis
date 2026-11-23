import torch

from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.model.interface.network_outputs import NetworkOutputs
from einn_multimodel.model.interface.phase_context import PhaseContext


class EINNForwardEngine:
    """EINN forward engine routing samples by true model in true-only mode."""

    def __init__(self, train_config: EINNTrainConfig, scaling_stats: DataScalingStats):
        self.config = train_config
        self.scaling_stats = scaling_stats

    @staticmethod
    def _get_jacobian_rows(outputs: torch.Tensor, inputs: torch.Tensor) -> list[torch.Tensor]:
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
        return torch.cat(self._get_jacobian_rows(outputs, time_tensor), dim=-1)

    def _compute_feature_gradient_trick(
        self,
        states: torch.Tensor,
        embedding: torch.Tensor,
        de_dt: torch.Tensor,
    ) -> torch.Tensor:
        ds_de_rows = self._get_jacobian_rows(states, embedding)
        return torch.cat(
            [(de_dt * row).sum(dim=-1, keepdim=True) for row in ds_de_rows], dim=-1
        )

    @staticmethod
    def _labels(context: PhaseContext) -> torch.Tensor:
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
        self, models, labels: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
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
        models,
        labels: torch.Tensor,
        states_real: torch.Tensor,
        detach_params: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        models,
        aux_mean: torch.Tensor,
        aux_std: torch.Tensor,
    ) -> torch.Tensor:
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

        # The expanding-window loader retains all `full_steps` input/target
        # time indices, even when the observed X prefix is shorter.  The
        # future ODE objectives must use NEW time indices after this grid.
        # Without this extension, ds_dt[:, full_steps:] is always empty and
        # ode_future_T / ode_future_F silently become exact zero.
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
            t_raw = torch.cat((t_raw, future_t_raw), dim=1)

        aux_mean = self.scaling_stats.aux_mean.to(t_raw.device)
        aux_std = self.scaling_stats.aux_std.to(t_raw.device)
        t_min = self.scaling_stats.t_min.to(t_raw.device)
        t_max = self.scaling_stats.t_max.to(t_raw.device)
        t_range = (t_max - t_min).clamp_min(1e-8)
        t_scaled = ((t_raw - t_min) / t_range).requires_grad_(True)

        # Use the TRUE label to select the Time Module and the ODE model.
        # All original EINN embeddings are passed directly to OutputModule.
        e_t_full = self._route_time_modules(models, labels, t_scaled)
        s_t_full = models.output_module(e=e_t_full)
        s_t_full = self._enforce_dummy_states(
            s_t_full, labels, models, aux_mean, aux_std
        )

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

        if context.phase_num >= 3:
            feature_outputs = models.feature_module(
                x=x_input,
                t=t_scaled.detach(),
                mask=context.mask,
                true_model_label=labels,
            )
            e_t_f_full = feature_outputs.embeddings
            if context.phase_num == 4:
                e_t_f_full = e_t_f_full.detach()
            s_t_f_full = models.output_module(e=e_t_f_full)
            s_t_f_full = self._enforce_dummy_states(
                s_t_f_full, labels, models, aux_mean, aux_std
            )

            out.e_t_F = e_t_f_full[:, :full_steps]
            out.s_t_F = s_t_f_full[:, :full_steps]

        if context.phase_num == 4:
            # Original EINN gradient trick on the unprojected embeddings.
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
