import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Multi-positive supervised contrastive loss.

    Supports [B, D] pooled features and [B, T, D] time-step features. For
    time-step features, the contrastive objective is computed independently at
    each time index over non-padding samples, then averaged over valid times.
    """

    def __init__(self, temperature: float = 0.1):
        """Initialize SupConLoss to evaluate timestep-level supervised contrastive loss over valid observation pairs.


        :param float temperature: Positive contrastive temperature controlling the logits’ scale.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def _loss_2d(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform loss 2d as part of evaluate timestep-level supervised contrastive loss over valid observation pairs.


        :param torch.Tensor features: The features used by this operation.
        :param torch.Tensor labels: Class identifiers assigned to the samples being compared.
        :return torch.Tensor: Tensor containing the computed model outputs or time derivatives.
        """
        batch_size = features.size(0)
        if batch_size < 2:
            return features.sum() * 0.0

        features = F.normalize(features, p=2, dim=-1)
        logits = (features @ features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        non_self = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positives = labels[:, None].eq(labels[None, :]) & non_self
        positive_count = positives.sum(dim=1)
        valid_anchor = positive_count > 0
        if not bool(valid_anchor.any()):
            return features.sum() * 0.0

        exp_logits = torch.exp(logits) * non_self.to(logits.dtype)
        log_denominator = torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
        )
        log_prob = logits - log_denominator
        mean_positive_log_prob = (
            (log_prob * positives.to(log_prob.dtype)).sum(dim=1)
            / positive_count.clamp_min(1)
        )
        return -mean_positive_log_prob[valid_anchor].mean()

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate evaluate timestep-level supervised contrastive loss over valid observation pairs for batched inputs.


        :param torch.Tensor features: The features used by this operation.
        :param torch.Tensor labels: Class identifiers assigned to the samples being compared.
        :param torch.Tensor | None mask: Observation mask; zero represents a valid time point and one padding.
        :return torch.Tensor: Tensor containing the computed model outputs or time derivatives.
        """
        labels = labels.reshape(-1).to(device=features.device).long()
        if features.ndim == 2:
            if labels.numel() != features.size(0):
                raise ValueError("labels must contain one value per feature vector.")
            return self._loss_2d(features, labels)
        if features.ndim != 3:
            raise ValueError("features must have shape [B, D] or [B, T, D].")
        if labels.numel() != features.size(0):
            raise ValueError("labels must contain one value per sequence.")
        if mask is not None and mask.shape != features.shape[:2]:
            raise ValueError("mask must have shape [B, T].")
        if mask is not None:
            mask = mask.to(device=features.device)

        time_losses = []
        for time_idx in range(features.size(1)):
            if mask is None:
                valid = torch.ones(
                    features.size(0), dtype=torch.bool, device=features.device
                )
            else:
                valid = ~mask[:, time_idx].bool()
            if int(valid.sum()) < 2:
                continue
            time_losses.append(
                self._loss_2d(features[valid, time_idx], labels[valid])
            )
        if not time_losses:
            return features.sum() * 0.0
        return torch.stack(time_losses).mean()
