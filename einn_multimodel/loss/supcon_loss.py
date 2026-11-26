import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Multi-positive supervised contrastive loss over pooled or timestep-level embeddings.

    Supports both 2D pooled embeddings of shape (B, D) and 3D sequential embeddings of
    shape (B, T, D). For 3D inputs, evaluates the contrastive objective independently at
    each timestep t over all non-padded batch samples and averages across valid timesteps
    (thesis Section 2.4).
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize SupConLoss with a positive similarity temperature tau.

        :param float temperature: Positive temperature scalar tau > 0 scaling pairwise cosine
            similarities; smaller values sharpen separation between positive and negative pairs.
        :return None: Stores temperature in place.
        :raises ValueError: If temperature is not strictly positive.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature: float = float(temperature)

    def _loss_2d(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the multi-positive SupCon loss for a single slice of embeddings (B, D).

        L2-normalizes embeddings internally to unit vectors z_i^F without modifying the
        caller's tensors (thesis Section 2.4), forms positive pairs P(i) = {j != i : a_j = a_i},
        and averages the negative mean positive log-probability over all anchors with |P(i)| > 0.

        :param torch.Tensor features: Embedding matrix of shape (B, D).
        :param torch.Tensor labels: Integer class labels a_i of shape (B,).
        :return torch.Tensor: Scalar SupCon loss of shape (); returns 0.0 (attached to graph)
            if B < 2 or no anchor has a positive pair.
        """
        batch_size = features.size(0)
        if batch_size < 2:
            return features.sum() * 0.0

        # Project embeddings onto the unit hypersphere: z = e / ||e||_2, shape (B, D).
        features = F.normalize(features, p=2, dim=-1)

        # Pairwise scaled cosine similarities <z_i, z_j> / tau of shape (B, B), stabilized by row max.
        logits = (features @ features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Exclude self-comparisons (diagonal) and identify same-class positive pairs P(i): shape (B, B).
        non_self = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positives = labels[:, None].eq(labels[None, :]) & non_self
        positive_count = positives.sum(dim=1)  # Shape: (B,)
        valid_anchor = positive_count > 0
        if not bool(valid_anchor.any()):
            return features.sum() * 0.0

        # Denominator sums exp(logit) over all k != i; compute log-softmax probabilities (B, B).
        exp_logits = torch.exp(logits) * non_self.to(logits.dtype)
        log_denominator = torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
        )
        log_prob = logits - log_denominator

        # Average log-probability over each anchor's positive set P(i), then negate and average over valid anchors.
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
        """
        Evaluate SupConLoss on 2D pooled (B, D) or 3D timestep-level (B, T, D) embeddings.

        For 3D inputs, computes _loss_2d independently at each timestep t over samples where
        mask[:, t] == 0 (observed), skipping timesteps with fewer than 2 valid samples, and
        returns the mean across all valid timesteps.

        :param torch.Tensor features: Latent embeddings of shape (B, D) or (B, T, D).
        :param torch.Tensor labels: True epidemiological model labels of shape (B,).
        :param torch.Tensor | None mask: Optional padding mask of shape (B, T) where 0 marks
            valid observed timesteps and 1 marks padded future timesteps; ignored for 2D inputs.
        :return torch.Tensor: Scalar SupCon loss of shape ().
        :raises ValueError: If features is not 2D/3D, labels length does not match B, or mask
            shape does not match (B, T).
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

        # Evaluate contrastive loss per timestep over non-padded samples (thesis Section 2.4).
        time_losses = []
        for time_idx in range(features.size(1)):
            if mask is None:
                valid = torch.ones(
                    features.size(0), dtype=torch.bool, device=features.device
                )
            else:
                valid = ~mask[:, time_idx].bool()  # True for observed (mask == 0) samples.
            if int(valid.sum()) < 2:
                continue
            time_losses.append(
                self._loss_2d(features[valid, time_idx], labels[valid])
            )

        if not time_losses:
            return features.sum() * 0.0
        return torch.stack(time_losses).mean()
