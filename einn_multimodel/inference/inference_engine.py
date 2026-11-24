"""
Feature-branch inference engine for trained multi-model EINN models.

Performs epidemiological state forecasting using only the shared FeatureModule and
OutputModule conditioned on a known model label, without invoking TimeModules or
ODEModules at inference time (thesis Section 2.4).
"""
from typing import Optional

import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.model.interface.einn_models import EINNModels


class InferenceEngine:
    """
    Inference engine executing the FeatureModule -> OutputModule forecasting path.

    Maps observed historical prefixes and target evaluation times to physical-domain
    epidemiological compartment trajectories, enforcing zero values on any dummy padding
    compartments introduced for multi-model state-dimension alignment (thesis Section 2.4).
    """

    def __init__(self, models: EINNModels) -> None:
        """
        Initialize the inference engine with a trained multi-model EINN container.

        :param EINNModels models: Container holding trained time_modules, feature_module,
            output_module, ode_models, and the model_to_id label registry.
        :return None: Stores the model container reference in place.
        """
        self.models: EINNModels = models

    def predict_future(
        self,
        historical_x: torch.Tensor,
        historical_x_mask: Optional[torch.Tensor],
        future_t: torch.Tensor,
        scaling_stats: DataScalingStats,
        assumed_model_label: Optional[torch.Tensor] = None,
        true_model_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forecast physical-domain compartment states from an observed prefix and known model label.

        No model-family classifier is trained; multi-model inference requires specifying the
        model label (thesis Section 2.4). The assumed_model_label argument is retained as a
        backward-compatible alias for true_model_label.

        :param torch.Tensor historical_x: Standardized observed input prefix of shape (B, T_obs, d_x).
        :param Optional[torch.Tensor] historical_x_mask: Optional padding mask of shape (B, T_obs)
            where 0.0 marks observed steps and 1.0 marks padded steps, or None if unpadded.
        :param torch.Tensor future_t: Unscaled evaluation time grid of shape (T_eval, 1) or (B, T_eval, 1).
        :param DataScalingStats scaling_stats: Training-fitted temporal bounds (t_min, t_max)
            and compartment statistics (aux_mean, aux_std) for physical-domain inverse scaling.
        :param Optional[torch.Tensor] assumed_model_label: Backward-compatible alias for
            true_model_label; scalar or shape (B,).
        :param Optional[torch.Tensor] true_model_label: Known epidemiological model ID(s);
            scalar or shape (B,). Required when len(models.model_to_id) > 1.
        :return torch.Tensor: Predicted physical-domain compartment trajectories of shape
            (B, T_eval, d_s) with model-specific dummy compartments zeroed out.
        :raises ValueError: If true_model_label and assumed_model_label conflict, if no label
            is provided in a multi-model setup, if label shape does not match B, or if an
            unknown model ID is supplied.
        """
        self._set_eval_mode()
        device = next(self.models.feature_module.parameters()).device

        # Move inputs to the FeatureModule device and ensure batched time tensor (B, T_eval, 1).
        historical_x = historical_x.to(device=device, dtype=torch.float32)
        historical_x_mask = (
            historical_x_mask.to(device) if historical_x_mask is not None else None
        )
        future_t = future_t.to(device=device, dtype=torch.float32)
        if future_t.ndim == 2:
            future_t = future_t.unsqueeze(0)

        batch_size = historical_x.size(0)

        # Resolve and validate the integer model-label tensor of shape (B,).
        if true_model_label is not None and assumed_model_label is not None:
            if not torch.equal(torch.as_tensor(true_model_label), torch.as_tensor(assumed_model_label)):
                raise ValueError("Conflicting true and assumed labels during inference.")
        model_label = true_model_label if true_model_label is not None else assumed_model_label
        if model_label is None:
            if len(self.models.model_to_id) != 1:
                raise ValueError(
                    "true_model_label is required for multi-model inference "
                    "because the model has no classifier."
                )
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            labels = torch.as_tensor(
                model_label, dtype=torch.long, device=device
            ).reshape(-1)
            if labels.numel() == 1:
                labels = labels.expand(batch_size)
            if labels.numel() != batch_size:
                raise ValueError("true_model_label must be scalar or [B].")
            if not set(labels.tolist()).issubset(set(self.models.model_to_id.values())):
                raise ValueError("Unknown true_model_label supplied.")

        # Min-max scale evaluation times using training bounds: (B, T_eval, 1) -> [0, 1].
        t_min = scaling_stats.t_min.to(device)
        t_max = scaling_stats.t_max.to(device)
        aux_mean = scaling_stats.aux_mean.to(device)
        aux_std = scaling_stats.aux_std.to(device)
        future_t_scaled = (future_t - t_min) / (t_max - t_min).clamp_min(1e-8)

        with torch.no_grad():
            # Encode historical prefix and decode timestep embeddings e_t^F of shape (B, T_eval, d_e).
            feature_output = self.models.feature_module(
                x=historical_x,
                t=future_t_scaled,
                mask=historical_x_mask,
                true_model_label=labels,
            )

            # Pass e_t^F directly to OutputModule (no projection head; thesis Section 2.4)
            # to obtain normalized states (B, T_eval, d_s), then invert Z-score scaling.
            states_normalized = self.models.output_module(
                e=feature_output.embeddings
            )
            states = states_normalized * aux_std + aux_mean
            return self._enforce_physical_dummy_states(states, labels)

    def _enforce_physical_dummy_states(
        self,
        states: torch.Tensor,
        model_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out structural padding compartments for models with smaller native state spaces.

        For example, in SIR-SEIR runs (thesis Section 2.4), SIR uses a constant-zero dummy E
        compartment to match SEIR's 4-state layout; this method enforces that exact zero value
        in physical space.

        :param torch.Tensor states: Predicted physical-domain states of shape (B, T_eval, d_s).
        :param torch.Tensor model_labels: Integer model IDs of shape (B,).
        :return torch.Tensor: Cloned state tensor of shape (B, T_eval, d_s) with dummy
            compartment indices set to 0.0 for matching samples.
        """
        result = states.clone()
        for name, model in self.models.ode_models.items():
            dummy_indices = getattr(model, "dummy_state_indices", ())
            if not dummy_indices:
                continue
            sample_indices = torch.where(
                model_labels == self.models.model_to_id[name]
            )[0]
            if sample_indices.numel() == 0:
                continue
            for state_idx in dummy_indices:
                result[sample_indices, :, state_idx] = 0.0
        return result

    def _set_eval_mode(self) -> None:
        """
        Switch all neural and mechanistic submodules to PyTorch evaluation mode.

        :return None: Updates module training flags in place.
        """
        self.models.time_modules.eval()
        self.models.feature_module.eval()
        self.models.output_module.eval()
        self.models.ode_models.eval()
