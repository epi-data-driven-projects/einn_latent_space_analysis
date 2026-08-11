import dataclasses
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from einn.config.einn_config import EINNConfig
from einn.model.interface.einn_models import EINNModels


class TrainingPostProcessor:
    """
    Handles I/O operations: saving model weights, hyperparameters, training metrics,
    and making model predictions persistent on the disk.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initializes the post-processor and creates the necessary output directory.

        Args:
            output_dir (str): Relative or absolute path to the output directory.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_and_save(
            self,
            models: EINNModels,
            config: EINNConfig,
            metrics_list: list,
            predictions: torch.Tensor,
            state_names: Optional[List[str]] = None
    ) -> None:
        """
        Saves the trained models, configuration, metrics, and predicted trajectories.
        Generalized to support any ODE model structure and dynamic batch sizes.

        Args:
            models (EINNModels): Container with the trained PyTorch models.
            config (EINNConfig): Configuration dataclass containing hyperparameters.
            metrics_list (list): List of TrainingMetrics dataclasses logged during training.
            predictions (torch.Tensor): Predicted future states. Expected shape: [Batch, Seq_len, d_s].
            state_names (Optional[List[str]]): Specific names for ODE states (e.g., ["S", "I", "R"]).
                                               Defaults to generic ["State_0", "State_1", ...] if None.
        """
        print(f"Saving results to directory: {self.output_dir} ...")

        # 1. Save neural network weights (.pt) - saved to CPU device format for portability
        torch.save(models.time_module.state_dict(), self.output_dir / "time_module.pt")
        torch.save(models.feature_module.state_dict(), self.output_dir / "feature_module.pt")
        torch.save(models.output_module.state_dict(), self.output_dir / "output_module.pt")

        # 2. Save hyperparameters (.json)
        config_dict = dataclasses.asdict(config)
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)

        # 3. Aggregate and save metrics (.csv)
        if metrics_list:
            metrics_df = pd.DataFrame([dataclasses.asdict(m) for m in metrics_list])
            metrics_df.to_csv(self.output_dir / "training_metrics.csv", index=False)

        # 4. Process and save predictions (.csv)
        # To handle Batch > 1, we flatten [Batch, Seq_len, d_s] into [Batch * Seq_len, d_s]
        d_s = predictions.shape[-1]

        # Ensure tensor is on CPU before converting to NumPy
        preds_np = predictions.view(-1, d_s).cpu().numpy()

        # Generate generic column names if specific names (like SEIRM) are not provided
        if state_names is None or len(state_names) != d_s:
            state_names = [f"State_{i}" for i in range(d_s)]

        preds_df = pd.DataFrame(preds_np, columns=state_names)
        preds_df.to_csv(self.output_dir / "predictions.csv", index=False)

        print("Save successful!")
