"""
Artifact serialization utility for trained multi-model EINN checkpoints, scalers, and metrics.

Persists model-specific and shared module weights, training-fitted scaling statistics, and
training histories so downstream evaluation and plotting can be reproduced without retraining
(thesis Section 3).
"""
import os
from typing import List

import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn.model.interface.training_metrics import TrainingMetrics
from einn_multimodel.model.interface.einn_models import EINNModels


class TrainingPostProcessor:
    """
    Persist trained multi-model EINN state dictionaries, label registries, scaling stats, and metrics.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initialize the post-processor and ensure the target output directory exists.

        :param str output_dir: Directory where serialized checkpoints and metadata are written.
        :return None: Stores output_dir and creates the directory on disk if missing.
        """
        self.output_dir: str = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_models(self, models: EINNModels, prefix: str = "einn") -> None:
        """
        Save all multi-model EINN submodule state dictionaries and the model-label registry.

        Writes five files into self.output_dir:
        <prefix>_time_modules.pth, <prefix>_ode_models.pth, <prefix>_feature_module.pth,
        <prefix>_output_module.pth, and <prefix>_model_to_id.pt.

        :param EINNModels models: Trained multi-model container holding time_modules, ode_models,
            feature_module, output_module, and model_to_id.
        :param str prefix: Filename prefix prepended to each saved checkpoint artifact.
        :return None: Writes serialized checkpoint files to self.output_dir.
        """
        # Save model-specific TimeModules and mechanistic ODE parameter modules.
        torch.save(models.time_modules.state_dict(), os.path.join(
            self.output_dir, f"{prefix}_time_modules.pth"
        ))
        torch.save(models.ode_models.state_dict(), os.path.join(
            self.output_dir, f"{prefix}_ode_models.pth"
        ))

        # Save shared FeatureModule and OutputModule weights.
        torch.save(models.feature_module.state_dict(), os.path.join(
            self.output_dir, f"{prefix}_feature_module.pth"
        ))
        torch.save(models.output_module.state_dict(), os.path.join(
            self.output_dir, f"{prefix}_output_module.pth"
        ))

        # Save the model-name to integer-ID mapping required to reload multi-model routing.
        torch.save(models.model_to_id, os.path.join(
            self.output_dir, f"{prefix}_model_to_id.pt"
        ))

    def save_scaling_stats(
        self,
        scaling_stats: DataScalingStats,
        filename: str = "scaling_stats.pt",
    ) -> None:
        """
        Serialize the training-fitted Z-score and temporal min-max scaling statistics.

        :param DataScalingStats scaling_stats: Compartment mean/std, temporal bounds, and
            normalized initial states fitted on the training split.
        :param str filename: Output filename inside self.output_dir.
        :return None: Writes scaling_stats to disk via torch.save.
        """
        torch.save(scaling_stats, os.path.join(self.output_dir, filename))

    def save_metrics(
        self,
        metrics: List[TrainingMetrics],
        filename: str = "training_metrics.pt",
    ) -> None:
        """
        Serialize the list of per-iteration or per-phase training metrics.

        :param List[TrainingMetrics] metrics: Recorded loss and optimization diagnostics from training.
        :param str filename: Output filename inside self.output_dir.
        :return None: Writes metrics to disk via torch.save.
        """
        torch.save(metrics, os.path.join(self.output_dir, filename))
