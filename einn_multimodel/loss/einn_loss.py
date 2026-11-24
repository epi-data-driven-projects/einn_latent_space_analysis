"""
Composite four-phase loss module for multi-model EINN with Supervised Contrastive learning.

Implements the full EINN objective from thesis Sections 2.2-2.4: data fitting, auxiliary
supervision, model-specific monotonicity, initial-condition penalties, ODE residuals
(in-window and extrapolation), Time-to-Feature knowledge distillation, and Phase 3
SupConLoss on Feature embeddings. Also supports optional loss masking for mismatched-model
control experiments (Section 5).
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.loss.supcon_loss import SupConLoss
from einn_multimodel.model.interface.einn_models import EINNModels
from einn_multimodel.model.interface.network_outputs import NetworkOutputs
from einn_multimodel.model.interface.phase_context import PhaseContext


class EINNLoss(nn.Module):
    """
    Phase-aware composite loss for multi-model EINN training.

    Activates the appropriate subset of loss terms for each of the four optimization phases
    (thesis Section 2.2) and adds Supervised Contrastive loss (SupConLoss) on timestep-level
    Feature embeddings e_t^F during Phase 3 (Section 2.4).
    """

    def __init__(
        self,
        config: EINNTrainConfig,
        models: EINNModels,
        scaling_stats: DataScalingStats,
    ):
        """
        Initialize the loss module, verify shared ODE dimensions, and instantiate SupConLoss.

        :param EINNTrainConfig config: Training configuration supplying loss_weights,
            supcon_temperature, and mask_mismatched_losses.
        :param EINNModels models: Multi-model EINN container exposing ode_models and model_to_id.
        :param DataScalingStats scaling_stats: Training-fitted scaling metadata providing
            normalized initial_states for L_IC.

        :return None: Initializes module attributes and SupConLoss in place.

        :raises ValueError: If paired ODE models disagree on target_idx or unified state dimension d_s.
        """
        super().__init__()
        self.config: EINNTrainConfig = config
        self.models: EINNModels = models
        self.weights: Dict[str, float] = config.loss_weights
        self.scaling_stats: DataScalingStats = scaling_stats

        # Verify all paired ODE models share a unified state dimension d_s and target index j*.
        target_indices = {model.target_idx for model in models.ode_models.values()}
        state_dims = {model.d_s for model in models.ode_models.values()}
        if len(target_indices) != 1:
            raise ValueError("All models must use the same target_idx in this experiment.")
        if len(state_dims) != 1:
            raise ValueError("All models must expose the same state dimension.")

        self.target_idx: int = next(iter(target_indices))
        self.supcon_loss: SupConLoss = SupConLoss(config.supcon_temperature)

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        """
        Create a zero scalar tensor attached to reference's autograd graph and device.

        :param torch.Tensor reference: Reference tensor supplying device, dtype, and graph connection.

        :return torch.Tensor: Scalar zero tensor of shape ().
        """
        return reference.sum() * 0.0

    def calc_data_loss(self, states: torch.Tensor | None, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute target-compartment MSE loss L_Data^T or L_Data^F (thesis Section 2.2).

        :param torch.Tensor | None states: Predicted compartment states of shape (B, T, d_s).
        :param torch.Tensor targets: Ground-truth target observations of shape (B, T, 1).

        :return torch.Tensor: Scalar MSE over the target compartment slice states[:, :, j*:j*+1].
        """
        if states is None:
            return targets.sum() * 0.0
        if states.numel() == 0 or targets.numel() == 0:
            return states.sum() * 0.0
        return F.mse_loss(
            states[:, :, self.target_idx:self.target_idx + 1], targets
        )

    @staticmethod
    def calc_aux_loss(states: torch.Tensor | None, aux_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary full-state supervision MSE loss L_aux (thesis Section 2.2).

        :param torch.Tensor | None states: Predicted Time-branch states of shape (B, T, d_s).
        :param torch.Tensor aux_targets: Standardized auxiliary reference states of shape (B, T, d_s).

        :return torch.Tensor: Scalar MSE across all compartments and timesteps.
        """
        if states is None or states.numel() == 0:
            return states.sum() * 0.0
        return F.mse_loss(states, aux_targets)

    @staticmethod
    def calc_ode_loss(ds_dt_nn: torch.Tensor | None, ds_dt_ode: torch.Tensor) -> torch.Tensor:
        """
        Compute mechanistic ODE residual MSE between neural and ODE time derivatives (Section 2.2).

        :param torch.Tensor | None ds_dt_nn: Autograd (or gradient-trick) time derivatives of shape (B, T, d_s).
        :param torch.Tensor ds_dt_ode: Mechanistic RHS evaluations f_ODE(s_hat, Omega) of shape (B, T, d_s).

        :return torch.Tensor: Scalar MSE ||ds_hat/dt - f_ODE(s_hat, Omega)||_2^2.
        """
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
        """
        Compute asymmetric ReLU-squared epidemiological monotonicity penalty L_Mono (Section 2.2).

        Penalizes negative derivatives on nondecreasing compartments (e.g. R) and positive
        derivatives on nonincreasing compartments (e.g. S in closed SIR/SEIR).

        :param torch.Tensor ds_dt: State time derivatives of shape (B, T, d_s).
        :param list[int] inc_indices: Compartment indices constrained to ds/dt >= 0.
        :param list[int] dec_indices: Compartment indices constrained to ds/dt <= 0.

        :return torch.Tensor: Scalar sum of mean squared monotonicity violations.
        """
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
        """
        Compute temporal first-difference smoothness penalty L_Param for time-varying ODE parameters.

        Returns 0.0 for constant global parameters (ndim < 3 or T <= 1; thesis Section 2.2).

        :param torch.Tensor params: ODE parameter tensor of shape (d_p,) or (B, T, d_p).

        :return torch.Tensor: Scalar mean squared temporal difference across consecutive timesteps.
        """
        if params.ndim < 3 or params.shape[1] <= 1:
            return params.sum() * 0.0
        return (params[:, 1:, :] - params[:, :-1, :]).square().mean()

    @staticmethod
    def calc_knowledge_distillation_loss(
        target: torch.Tensor,
        prediction: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute Time-to-Feature knowledge distillation MSE (L_KD,target or L_KD,emb; Section 2.2).

        :param torch.Tensor target: Detached Time-branch reference tensor of shape (B, T, d_s) or (B, T, d_e).
        :param torch.Tensor | None prediction: Feature-branch tensor of matching shape.

        :return torch.Tensor: Scalar MSE between Feature predictions and detached Time targets.
        """
        if prediction is None:
            return target.sum() * 0.0
        if prediction.numel() == 0:
            return prediction.sum() * 0.0
        return F.mse_loss(prediction, target)

    @staticmethod
    def calc_ic_loss(
        states: torch.Tensor | None,
        t: torch.Tensor,
        initial_states: list[float],
    ) -> torch.Tensor:
        """
        Compute initial-condition MSE penalty L_IC at t = 0 in normalized state space (Section 2.3).

        :param torch.Tensor | None states: Predicted compartment states of shape (B, T, d_s).
        :param torch.Tensor t: Time coordinates of shape (B, T, 1).
        :param list[float] initial_states: Normalized initial state vector of length d_s.

        :return torch.Tensor: Scalar MSE over all batch elements where t < 1e-3, or 0.0 if none.
        """
        if states is None or states.numel() == 0 or t.numel() == 0 or not initial_states:
            return states.sum() * 0.0
        mask_t0 = (t < 1e-3).squeeze(-1)  # Shape: (B, T)
        if not bool(mask_t0.any()):
            return states.sum() * 0.0
        states_at_t0 = states[mask_t0]    # Shape: (N_t0, d_s)
        target_ic = torch.as_tensor(
            initial_states, dtype=states.dtype, device=states.device
        ).unsqueeze(0).expand(states_at_t0.shape[0], -1)
        return F.mse_loss(states_at_t0, target_ic)

    def _monotonicity_by_assumption(
        self,
        derivatives: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate model-specific monotonicity penalties grouped by assumed model label.

        Weights each model family's monotonicity loss by its fraction of samples in the batch
        so the total penalty remains scale-invariant across batch compositions.

        :param torch.Tensor derivatives: ODE time derivatives of shape (B, T, d_s).
        :param torch.Tensor labels: Assumed integer model IDs of shape (B,).

        :return torch.Tensor: Scalar batch-weighted monotonicity penalty.
        """
        losses = []
        total = max(1, derivatives.shape[0])
        for name, model in self.models.ode_models.items():
            model_id = self.models.model_to_id[name]
            indices = torch.where(labels == model_id)[0]
            if indices.numel() == 0:
                continue
            subset = derivatives.index_select(0, indices)  # Shape: (B_model, T, d_s)
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
        """
        Compute the weighted total loss L_total and per-component breakdown for the active phase.

        Phase activation schedule (thesis Sections 2.2-2.4):
        Phase >= 1: data_T, aux, mono, ic.
        Phase >= 2: ode_T, ode_future_T, param.
        Phase >= 3: data_F, kd_target, kd_emb, ic_F (plus supcon exclusively in Phase 3).
        Phase == 4: ode_F, ode_future_F (via gradient trick).

        :param PhaseContext phase_context: Current phase index, batch inputs (x, y, t, aux_targets,
            mask), and model labels (true_model_label, assumed_model_label).
        :param NetworkOutputs network_outputs: Forward-pass states, embeddings (e_t, e_t_F),
            neural/ODE derivatives, and ODE parameters.

        :return tuple[torch.Tensor, Dict[str, float]]: Pair (total_loss, loss_dict) containing
            the differentiable scalar objective and detached weighted component values.

        :raises ValueError: If required model labels are missing or parameter batch shapes mismatch.
        """
        total_loss = phase_context.t.new_zeros(())
        loss_dict: Dict[str, float] = {}

        def add_loss(name: str, value: torch.Tensor) -> None:
            """
            Accumulate weight * value into total_loss and log its detached scalar if weight > 0.

            :param str name: Loss key in self.weights.
            :param torch.Tensor value: Unweighted scalar loss tensor.

            :return None: Updates total_loss and loss_dict in place.
            """
            nonlocal total_loss
            weight = float(self.weights.get(name, 0.0))
            if weight > 0:
                weighted = weight * value
                total_loss += weighted
                loss_dict[name] = float(weighted.detach().cpu())

        # Resolve assumed model labels of shape (B,) for routing Time/ODE/monotonicity terms.
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

        # Optional boolean mask (B,) restricting physics/Time/KD losses to correctly specified
        # samples (true_label == assumed_label) before batch reduction.
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
            """
            Filter a batched tensor along dim 0 using the matched mask when masking is enabled.

            :param torch.Tensor | None tensor: Batched tensor of shape (B, ...) or None.

            :return torch.Tensor | None: Filtered tensor of shape (B_matched, ...) or original tensor.
            """
            if tensor is None or matched is None:
                return tensor
            return tensor[matched]

        # Phase 1+: Time-branch data fitting, auxiliary supervision, monotonicity, and t=0 IC loss.
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

        # Phase 2+: Mechanistic ODE residuals on Time branch (in-window and future) and parameter smoothness.
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
                for params in (network_outputs.params or {}).values():
                    add_loss("param", self.calc_parameter_smoothness_loss(params))
            else:
                # Mask per-trajectory parameters [B_model, T, d_p] by matched samples;
                # keep global [d_p] or broadcast [1, T, d_p] parameters when at least one sample matches.
                parameter_losses = []
                for model_name, params in (network_outputs.params or {}).items():
                    group_id = self.models.model_to_id[model_name]
                    group_matched = matched[labels == group_id]

                    if not bool(group_matched.any()):
                        continue

                    if params.ndim < 3:
                        parameter_losses.append(
                            self.calc_parameter_smoothness_loss(params)
                        )
                    elif params.shape[0] == group_matched.numel():
                        parameter_losses.append(
                            self.calc_parameter_smoothness_loss(params[group_matched])
                        )
                    elif params.shape[0] == 1:
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

        # Phase 3+: Feature-branch data/IC losses (all samples), Time-to-Feature KD, and Phase 3 SupCon.
        if phase_context.phase_num >= 3:
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
                # SupCon operates on all samples using true epidemiological model classes (Section 2.4).
                add_loss("supcon", self.supcon_loss(
                    network_outputs.e_t_F,
                    true_labels,
                    mask=phase_context.mask,
                ))

        # Phase 4: Feature-branch ODE residuals via gradient trick (in-window and extrapolation).
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
