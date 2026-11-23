from typing import Optional

import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.model.interface.einn_models import EINNModels


class InferenceEngine:
    """Inference using only the Feature Module and shared Output Module."""

    def __init__(self, models: EINNModels):
        """Initialize InferenceEngine to predict future states from the observed prefix using the trained Feature module.


        :param EINNModels models: The models containing the time, feature, output, and ODE modules.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.models = models

    def predict_future(
        self,
        historical_x: torch.Tensor,
        historical_x_mask: Optional[torch.Tensor],
        future_t: torch.Tensor,
        scaling_stats: DataScalingStats,
        assumed_model_label: Optional[torch.Tensor] = None,
        true_model_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Infer with FeatureModule only, supplied with the KNOWN true class.

        No classifier is trained. Unknown model classes cannot be inferred.
        The assumed_model_label kw remains for backwards compatibility.

        :param torch.Tensor historical_x: Observed feature prefix from which future predictions are generated.
        :param Optional[torch.Tensor] historical_x_mask: Mask indicating the valid time points of the observed feature prefix.
        :param torch.Tensor future_t: Requested evaluation times for reconstruction and forecasting.
        :param DataScalingStats scaling_stats: Statistics fitted on training data for physical-domain conversion.
        :param Optional[torch.Tensor] assumed_model_label: Identifier of the model hypothesis supplied to the network.
        :param Optional[torch.Tensor] true_model_label: Identifier of the data-generating mechanistic model.
        :return torch.Tensor: Tensor containing the computed model outputs or time derivatives.
        """
        self._set_eval_mode()
        device = next(self.models.feature_module.parameters()).device
        historical_x = historical_x.to(device=device, dtype=torch.float32)
        historical_x_mask = (
            historical_x_mask.to(device) if historical_x_mask is not None else None
        )
        future_t = future_t.to(device=device, dtype=torch.float32)
        if future_t.ndim == 2:
            future_t = future_t.unsqueeze(0)

        batch_size = historical_x.size(0)
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

        t_min = scaling_stats.t_min.to(device)
        t_max = scaling_stats.t_max.to(device)
        aux_mean = scaling_stats.aux_mean.to(device)
        aux_std = scaling_stats.aux_std.to(device)
        future_t_scaled = (future_t - t_min) / (t_max - t_min).clamp_min(1e-8)

        with torch.no_grad():
            feature_output = self.models.feature_module(
                x=historical_x,
                t=future_t_scaled,
                mask=historical_x_mask,
                true_model_label=labels,
            )
            # The SAME e_t^F (optionally L2-normalized by FeatureModule)
            # goes directly to OutputModule, with no projection head.
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
        """Perform enforce physical dummy states as part of predict future states from the observed prefix using the trained Feature module.


        :param torch.Tensor states: Current compartment populations in model state order.
        :param torch.Tensor model_labels: Epidemiological model identifiers assigned to each input trajectory.
        :return torch.Tensor: Tensor containing the computed model outputs or time derivatives.
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
        """Perform set eval mode as part of predict future states from the observed prefix using the trained Feature module.


        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.models.time_modules.eval()
        self.models.feature_module.eval()
        self.models.output_module.eval()
        self.models.ode_models.eval()
