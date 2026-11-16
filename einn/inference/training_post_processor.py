import os
from typing import List

import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn.model.interface.einn_models import EINNModels
from einn.model.interface.training_metrics import TrainingMetrics


class TrainingPostProcessor:
    """
    Handles the serialization of model artifacts, scaling statistics, and training metrics
    after the EINN training loop concludes.

    This ensures that the trained components are safely exported to disk in a clean,
    framework-agnostic binary format (.pth/.pt) for independent evaluation or plotting later.
    """

    def __init__(self, output_dir: str):
        """
        Initializes the post-processor and ensures the designated output directory exists.

        :param str output_dir: The directory path where models and statistics will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_models(self, models: EINNModels, prefix: str = "einn"):
        """
        Extracts and saves the PyTorch state dictionaries of all trained neural network modules.

        :param EINNModels models: The dataclass containing the trained Time, Feature, and Output modules.
        :param str prefix: A string prefix prepended to the saved files (default: "einn").
        """
        time_path = os.path.join(self.output_dir, f"{prefix}_time_module.pth")
        feature_path = os.path.join(self.output_dir, f"{prefix}_feature_module.pth")
        output_path = os.path.join(self.output_dir, f"{prefix}_output_module.pth")

        # Serializing only the learned weights (state_dict) is much safer and more robust
        # than serializing the entire Python class objects.
        torch.save(models.time_module.state_dict(), time_path)
        torch.save(models.feature_module.state_dict(), feature_path)
        torch.save(models.output_module.state_dict(), output_path)

        print(f"Neural network weights successfully saved to {self.output_dir}.")

    def save_scaling_stats(self, scaling_stats: DataScalingStats, filename: str = "scaling_stats.pt"):
        """
        Serializes the DataScalingStats dataclass.

        This is critical for the independent Evaluator, as it needs the exact means and
        standard deviations used during training to inverse-transform the neural predictions
        back into the biological domain (e.g., actual population fractions).

        :param DataScalingStats scaling_stats: The runtime scaling statistics from the preprocessing phase.
        :param str filename: The name of the output binary file (default: "scaling_stats.pt").
        """
        stats_path = os.path.join(self.output_dir, filename)
        torch.save(scaling_stats, stats_path)

        print(f"Scaling statistics successfully saved to {stats_path}.")

    def save_metrics(self, metrics: List[TrainingMetrics], filename: str = "training_metrics.pt"):
        """
        Saves the raw training metrics list for later numerical analysis or plotting.

        :param List[TrainingMetrics] metrics: The list of dataclasses collected by the Trainer per epoch/phase.
        :param str filename: The name of the output binary file (default: "training_metrics.pt").
        """
        metrics_path = os.path.join(self.output_dir, filename)
        torch.save(metrics, metrics_path)

        print(f"Training metrics successfully saved to {metrics_path}.")
