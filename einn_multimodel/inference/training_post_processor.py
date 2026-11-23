import os
from typing import List

import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.model.interface.einn_models import EINNModels
from einn.model.interface.training_metrics import TrainingMetrics


class TrainingPostProcessor:
    """Save model-specific and shared EINN module state dictionaries."""

    def __init__(self, output_dir: str):
        """Initialize TrainingPostProcessor to save trained weights, scaling statistics, and training metrics.


        :param str output_dir: Directory where generated results are written.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_models(self, models: EINNModels, prefix: str = "einn") -> None:
        """Persist save models using the original serialization format.


        :param EINNModels models: The models containing the time, feature, output, and ODE modules.
        :param str prefix: Prefix used to indent descendants of a hierarchical tree node.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        torch.save(models.time_modules.state_dict(), os.path.join(
            self.output_dir, f"{prefix}_time_modules.pth"
        ))
        torch.save(models.ode_models.state_dict(), os.path.join(
            self.output_dir, f"{prefix}_ode_models.pth"
        ))
        torch.save(models.feature_module.state_dict(), os.path.join(
            self.output_dir, f"{prefix}_feature_module.pth"
        ))
        torch.save(models.output_module.state_dict(), os.path.join(
            self.output_dir, f"{prefix}_output_module.pth"
        ))
        torch.save(models.model_to_id, os.path.join(
            self.output_dir, f"{prefix}_model_to_id.pt"
        ))

    def save_scaling_stats(
            self,
            scaling_stats: DataScalingStats,
            filename: str = "scaling_stats.pt",
    ) -> None:
        """Persist save scaling stats using the original serialization format.


        :param DataScalingStats scaling_stats: Statistics fitted on training data for physical-domain conversion.
        :param str filename: Path or name of a saved artifact.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        torch.save(scaling_stats, os.path.join(self.output_dir, filename))

    def save_metrics(
            self,
            metrics: List[TrainingMetrics],
            filename: str = "training_metrics.pt",
    ) -> None:
        """Persist save metrics using the original serialization format.


        :param List[TrainingMetrics] metrics: Training, forecast, or latent-space diagnostic measurements.
        :param str filename: Path or name of a saved artifact.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        torch.save(metrics, os.path.join(self.output_dir, filename))
