from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.loss.supcon_loss import SupConLoss
from einn_multimodel.model.interface.network_outputs import NetworkOutputs
from einn_multimodel.model.interface.phase_context import PhaseContext


class EINNLoss(nn.Module):
    """Original four-phase EINN losses plus Feature-embedding SupCon."""

    def __init__(
        self,
        config: EINNTrainConfig,
        models,
        scaling_stats: DataScalingStats,
    ):
        super().__init__()
        self.config = config
        self.models = models
        self.weights = config.loss_weights
        self.scaling_stats = scaling_stats
        target_indices = {model.target_idx for model in models.ode_models.values()}
        state_dims = {model.d_s for model in models.ode_models.values()}
        if len(target_indices) != 1:
            raise ValueError("All models must use the same target_idx in this experiment.")
        if len(state_dims) != 1:
            raise ValueError("All models must expose the same state dimension.")
        self.target_idx = next(iter(target_indices))
        self.supcon_loss = SupConLoss(config.supcon_temperature)

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def calc_data_loss(self, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if states is None:
            return targets.sum() * 0.0
        if states.numel() == 0 or targets.numel() == 0:
            return states.sum() * 0.0
        return F.mse_loss(
            states[:, :, self.target_idx:self.target_idx + 1], targets
        )

    @staticmethod
    def calc_aux_loss(states: torch.Tensor, aux_targets: torch.Tensor) -> torch.Tensor:
        if states is None or states.numel() == 0:
            return states.sum() * 0.0
        return F.mse_loss(states, aux_targets)

    @staticmethod
    def calc_ode_loss(ds_dt_nn: torch.Tensor, ds_dt_ode: torch.Tensor) -> torch.Tensor:
        if ds_dt_nn is None:
            return ds_dt_ode.sum() * 0.0
        if ds_dt_nn.numel() == 0:
            return ds_dt_nn.sum() * 0.0
        return F.mse_loss(ds_dt_nn, ds_dt_ode)

    @staticmethod
    def calc_monotonicity_loss(
        ds_dt: torch.Tensor,
        inc_indices: list[int],
        dec_indices: list[int],
    ) -> torch.Tensor:
        if ds_dt.numel() == 0:
            return ds_dt.sum() * 0.0
        penalties = []
        for idx in inc_indices:
            val = -ds_dt[:, :, idx]
            penalties.append((val * torch.relu(val)).square().mean())
        for idx in dec_indices:
            val = ds_dt[:, :, idx]
            penalties.append((val * torch.relu(val)).square().mean())
        return torch.stack(penalties).sum() if penalties else ds_dt.sum() * 0.0

    @staticmethod
    def calc_parameter_smoothness_loss(params: torch.Tensor) -> torch.Tensor:
        if params.ndim < 3 or params.shape[1] <= 1:
            return params.sum() * 0.0
        return (params[:, 1:, :] - params[:, :-1, :]).square().mean()

    @staticmethod
    def calc_knowledge_distillation_loss(
        target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        if prediction is None:
            return target.sum() * 0.0
        if prediction.numel() == 0:
            return prediction.sum() * 0.0
        return F.mse_loss(prediction, target)

    @staticmethod
    def calc_ic_loss(
        states: torch.Tensor,
        t: torch.Tensor,
        initial_states: list[float],
    ) -> torch.Tensor:
        if states is None or states.numel() == 0 or t.numel() == 0 or not initial_states:
            return states.sum() * 0.0
        mask_t0 = (t < 1e-3).squeeze(-1)
        if not bool(mask_t0.any()):
            return states.sum() * 0.0
        states_at_t0 = states[mask_t0]
        target_ic = torch.as_tensor(
            initial_states, dtype=states.dtype, device=states.device
        ).unsqueeze(0).expand(states_at_t0.shape[0], -1)
        return F.mse_loss(states_at_t0, target_ic)

    def _monotonicity_by_assumption(
        self,
        derivatives: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        total = max(1, derivatives.shape[0])
        for name, model in self.models.ode_models.items():
            model_id = self.models.model_to_id[name]
            indices = torch.where(labels == model_id)[0]
            if indices.numel() == 0:
                continue
            subset = derivatives.index_select(0, indices)
            mono = model.monotonicity_indices
            losses.append(
                self.calc_monotonicity_loss(
                    subset,
                    inc_indices=mono.get("increasing", []),
                    dec_indices=mono.get("decreasing", []),
                ) * (indices.numel() / total)
            )
        return torch.stack(losses).sum() if losses else derivatives.sum() * 0.0

    def forward(
        self,
        phase_context: PhaseContext,
        network_outputs: NetworkOutputs,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        total_loss = phase_context.t.new_zeros(())
        loss_dict: Dict[str, float] = {}

        def add_loss(name: str, value: torch.Tensor) -> None:
            nonlocal total_loss
            weight = float(self.weights.get(name, 0.0))
            if weight > 0:
                weighted = weight * value
                total_loss = total_loss + weighted
                loss_dict[name] = float(weighted.detach().cpu())

        labels = phase_context.assumed_model_label
        if labels is None:
            if len(self.models.ode_models) != 1:
                raise ValueError(
                    "assumed_model_label is required for multi-model loss calculation."
                )
            labels = torch.zeros(
                phase_context.x.size(0), dtype=torch.long, device=phase_context.x.device
            )
        labels = labels.reshape(-1).long()

        # Restrict only physically model-dependent and knowledge-distillation
        # objectives to correctly specified samples. This happens BEFORE each
        # objective reduces its tensor over the batch.
        # None means legacy behavior: all samples remain active.
        matched: torch.Tensor | None = None
        if self.config.mask_mismatched_losses:
            true_labels = phase_context.true_model_label
            if true_labels is None:
                raise ValueError(
                    "true_model_label is required when mask_mismatched_losses=True."
                )
            true_labels = true_labels.reshape(-1).to(labels.device).long()
            if true_labels.shape != labels.shape:
                raise ValueError("true_model_label and assumed_model_label must match in shape.")
            matched = true_labels == labels

        def restrict(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or matched is None:
                return tensor
            return tensor[matched]

        if phase_context.phase_num >= 1:
            add_loss("data_T", self.calc_data_loss(
                restrict(network_outputs.s_t), restrict(phase_context.y)
            ))
            add_loss("aux", self.calc_aux_loss(
                restrict(network_outputs.s_t), restrict(phase_context.aux_targets)
            ))
            add_loss(
                "mono",
                self._monotonicity_by_assumption(
                    restrict(network_outputs.ds_dt_T_ode),
                    labels if matched is None else labels[matched],
                ),
            )
            add_loss(
                "ic",
                self.calc_ic_loss(
                    restrict(network_outputs.s_t),
                    restrict(phase_context.t[:, :network_outputs.s_t.shape[1]]),
                    self.scaling_stats.initial_states,
                ),
            )

        if phase_context.phase_num >= 2:
            add_loss("ode_T", self.calc_ode_loss(
                restrict(network_outputs.ds_dt_T_nn),
                restrict(network_outputs.ds_dt_T_ode),
            ))
            add_loss("ode_future_T", self.calc_ode_loss(
                restrict(network_outputs.ds_dt_future_T_nn),
                restrict(network_outputs.ds_dt_future_T_ode),
            ))
            if matched is None:
                # Preserve the old loss implementation when the switch is off.
                for params in (network_outputs.params or {}).values():
                    add_loss("param", self.calc_parameter_smoothness_loss(params))
            else:
                # Each ODE model may expose either global parameters [d_p]
                # (as in the current SIR/SIS and SIR/SEIR models) or
                # per-trajectory parameters [B_model, T, d_p]. Only the latter
                # have a batch axis that can be selected with a sample mask.
                parameter_losses = []
                for model_name, params in (network_outputs.params or {}).items():
                    group_id = self.models.model_to_id[model_name]
                    group_matched = matched[labels == group_id]

                    # No correctly specified example supports this ODE model
                    # in the current batch: omit its regularization term.
                    if not bool(group_matched.any()):
                        continue

                    if params.ndim < 3:
                        # Global parameters, e.g. [beta, gamma], are shared by
                        # every sample using the given ODE model. There is no
                        # per-sample axis to mask. Keep legacy loss semantics.
                        parameter_losses.append(
                            self.calc_parameter_smoothness_loss(params)
                        )
                    elif params.shape[0] == group_matched.numel():
                        # Per-trajectory, assumed-model-grouped parameters.
                        parameter_losses.append(
                            self.calc_parameter_smoothness_loss(params[group_matched])
                        )
                    elif params.shape[0] == 1:
                        # Model-wide, time-dependent parameters broadcast to
                        # the whole assumed-model group.
                        parameter_losses.append(
                            self.calc_parameter_smoothness_loss(params)
                        )
                    else:
                        raise ValueError(
                            f"Parameter batch alignment mismatch for {model_name}: "
                            f"shape={tuple(params.shape)}, "
                            f"assumed-model batch={group_matched.numel()}."
                        )

                if parameter_losses:
                    add_loss("param", torch.stack(parameter_losses).sum())
                else:
                    add_loss("param", self._zero(network_outputs.s_t))

        if phase_context.phase_num >= 3:
            # These Feature objectives deliberately use ALL samples, including
            # wrong assumptions: their true model label is still informative.
            add_loss("data_F", self.calc_data_loss(network_outputs.s_t_F, phase_context.y))
            add_loss("kd_target", self.calc_knowledge_distillation_loss(
                restrict(network_outputs.s_t.detach()),
                restrict(network_outputs.s_t_F),
            ))
            add_loss("kd_emb", self.calc_knowledge_distillation_loss(
                restrict(network_outputs.e_t.detach()),
                restrict(network_outputs.e_t_F),
            ))
            add_loss(
                "ic_F",
                self.calc_ic_loss(
                    network_outputs.s_t_F,
                    phase_context.t[:, :network_outputs.s_t_F.shape[1]],
                    self.scaling_stats.initial_states,
                ),
            )
            if phase_context.phase_num == 3:
                true_labels = phase_context.true_model_label
                if true_labels is None:
                    if len(self.models.ode_models) != 1:
                        raise ValueError(
                            "true_model_label is required for multi-model SupCon."
                        )
                    true_labels = torch.zeros(
                        phase_context.x.size(0), dtype=torch.long,
                        device=phase_context.x.device,
                    )
                true_labels = true_labels.reshape(-1).long()
                # SupCon remains on ALL samples and uses the true model class.
                add_loss("supcon", self.supcon_loss(
                    network_outputs.e_t_F,
                    true_labels,
                    mask=phase_context.mask,
                ))

        if phase_context.phase_num == 4:
            add_loss("ode_F", self.calc_ode_loss(
                restrict(network_outputs.ds_dt_F_nn),
                restrict(network_outputs.ds_dt_F_ode),
            ))
            add_loss("ode_future_F", self.calc_ode_loss(
                restrict(network_outputs.ds_dt_future_F_nn),
                restrict(network_outputs.ds_dt_future_F_ode),
            ))

        return total_loss, loss_dict
