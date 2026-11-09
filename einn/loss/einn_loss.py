import torch
import torch.nn as nn
import torch.nn.functional as F

from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext
from einn.ode.base_ode_model import BaseODEModel


class EINNLoss(nn.Module):
    """Masked composite loss for the expanding-prefix EINN windows."""

    def __init__(
        self,
        config: EINNTrainConfig,
        ode_model: BaseODEModel,
        scaling_stats: DataScalingStats,
    ) -> None:
        super().__init__()
        self.config = config
        self.ode_model = ode_model
        self.weights = config.loss_weights
        self.scaling_stats = scaling_stats
        self.mono_dec_indices = ode_model.monotonicity_indices["decreasing"]
        self.mono_inc_indices = ode_model.monotonicity_indices["increasing"]

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}."
            )
        if prediction.numel() == 0:
            return prediction.new_zeros(())
        squared_error = (prediction - target).square()
        if mask is None:
            return squared_error.mean()
        valid = mask.to(device=prediction.device, dtype=prediction.dtype)
        if valid.ndim == 2:
            valid = valid.unsqueeze(-1)
        valid = valid.expand_as(squared_error)
        denominator = valid.sum().clamp_min(1.0)
        return (squared_error * valid).sum() / denominator

    def calc_data_loss(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_idx = self.ode_model.target_idx
        prediction = states[:, :, target_idx : target_idx + 1]
        return self._masked_mse(prediction, targets, mask)

    @classmethod
    def calc_aux_loss(
        cls,
        states: torch.Tensor,
        aux_targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return cls._masked_mse(states, aux_targets, mask)

    @classmethod
    def calc_ode_loss(
        cls,
        ds_dt_nn: torch.Tensor,
        ds_dt_ode: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return cls._masked_mse(ds_dt_nn, ds_dt_ode, mask)

    @staticmethod
    def calc_monotonicity_loss(
        ds_dt: torch.Tensor,
        inc_indices: list[int],
        dec_indices: list[int],
    ) -> torch.Tensor:
        penalty = ds_dt.new_zeros(())
        for idx in inc_indices:
            violation = torch.relu(-ds_dt[:, :, idx])
            penalty = penalty + violation.square().mean()
        for idx in dec_indices:
            violation = torch.relu(ds_dt[:, :, idx])
            penalty = penalty + violation.square().mean()
        return penalty

    @staticmethod
    def calc_parameter_smoothness_loss(params: torch.Tensor) -> torch.Tensor:
        if params.ndim < 2 or params.shape[1] <= 1:
            return params.new_zeros(())
        return (params[:, 1:, :] - params[:, :-1, :]).square().mean()

    @staticmethod
    def calc_knowledge_distillation_loss(
        target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(prediction, target)

    @staticmethod
    def calc_ic_loss(
        states: torch.Tensor,
        t: torch.Tensor,
        initial_states: list[float],
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_t0 = (t.abs() < 1e-6).squeeze(-1)
        if target_mask is not None:
            mask_t0 = mask_t0 & (target_mask > 0)
        if not mask_t0.any():
            return states.new_zeros(())
        states_at_t0 = states[mask_t0]
        target_ic = torch.as_tensor(
            initial_states,
            dtype=states.dtype,
            device=states.device,
        ).unsqueeze(0).expand_as(states_at_t0)
        return F.mse_loss(states_at_t0, target_ic)

    def forward(
        self,
        phase_context: PhaseContext,
        network_outputs: NetworkOutputs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = phase_context.t.new_zeros(())
        loss_dict: dict[str, float] = {}
        target_mask = phase_context.target_mask
        history_steps = (
            phase_context.history_steps
            if phase_context.history_steps is not None
            else phase_context.t.shape[1]
        )
        history_steps = max(0, min(int(history_steps), phase_context.t.shape[1]))
        history_mask = (
            target_mask[:, :history_steps]
            if target_mask is not None
            else None
        )

        def add_loss(name: str, value: torch.Tensor) -> None:
            nonlocal total_loss
            weight = float(self.weights.get(name, 0.0))
            if weight > 0:
                weighted = weight * value
                total_loss = total_loss + weighted
                loss_dict[name] = float(weighted.detach().cpu())

        if phase_context.phase_num >= 1:
            add_loss(
                "data_T",
                self.calc_data_loss(
                    network_outputs.s_t,
                    phase_context.y,
                    target_mask,
                ),
            )
            add_loss(
                "aux",
                self.calc_aux_loss(
                    network_outputs.s_t,
                    phase_context.aux_targets,
                    target_mask,
                ),
            )
            # The model-generated time grid includes valid forecast times too.
            add_loss(
                "mono",
                self.calc_monotonicity_loss(
                    network_outputs.ds_dt_T_ode,
                    self.mono_inc_indices,
                    self.mono_dec_indices,
                ),
            )
            add_loss(
                "ic",
                self.calc_ic_loss(
                    network_outputs.s_t,
                    phase_context.t,
                    self.scaling_stats.initial_states,
                    target_mask,
                ),
            )

        if phase_context.phase_num >= 2:
            add_loss(
                "ode_T",
                self.calc_ode_loss(
                    network_outputs.ds_dt_T_nn[:, :history_steps, :],
                    network_outputs.ds_dt_T_ode[:, :history_steps, :],
                    history_mask,
                ),
            )
            add_loss(
                "ode_future_T",
                self.calc_ode_loss(
                    network_outputs.ds_dt_future_T_nn,
                    network_outputs.ds_dt_future_T_ode,
                ),
            )
            add_loss(
                "param",
                self.calc_parameter_smoothness_loss(network_outputs.params),
            )

        if phase_context.phase_num >= 3:
            add_loss(
                "data_F",
                self.calc_data_loss(
                    network_outputs.s_t_F,
                    phase_context.y,
                    target_mask,
                ),
            )
            add_loss(
                "kd_target",
                self.calc_knowledge_distillation_loss(
                    network_outputs.s_t.detach(),
                    network_outputs.s_t_F,
                ),
            )
            add_loss(
                "kd_emb",
                self.calc_knowledge_distillation_loss(
                    network_outputs.e_t.detach(),
                    network_outputs.e_t_F,
                ),
            )
            add_loss(
                "ic_F",
                self.calc_ic_loss(
                    network_outputs.s_t_F,
                    phase_context.t,
                    self.scaling_stats.initial_states,
                    target_mask,
                ),
            )

        if phase_context.phase_num == 4:
            add_loss(
                "ode_F",
                self.calc_ode_loss(
                    network_outputs.ds_dt_F_nn[:, :history_steps, :],
                    network_outputs.ds_dt_F_ode[:, :history_steps, :],
                    history_mask,
                ),
            )
            add_loss(
                "ode_future_F",
                self.calc_ode_loss(
                    network_outputs.ds_dt_future_F_nn,
                    network_outputs.ds_dt_future_F_ode,
                ),
            )

        return total_loss, loss_dict
