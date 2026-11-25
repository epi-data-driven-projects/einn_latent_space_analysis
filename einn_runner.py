from typing import Any

import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.inference.training_post_processor import TrainingPostProcessor
from einn.model.einn_builder import EINNBuilder
from einn.training.einn_trainer import EINNTrainer
from einn.utils.experiment_utils import ExperimentUtils


class EINNRunner:
    """
    Sets global seeds, instantiates `TrainingPostProcessor`, builds the requested single-model
    EINN architecture and target compartment index, executes the multiphase training loop, and
    exports serialized checkpoints, training metrics, and scaling statistics.
    """

    def __init__(
        self,
        output_dir: str,
        model_cfg: EINNModelConfig,
        train_cfg: EINNTrainConfig,
        seed: int | None = 42,
    ):
        """
        Initialize the runner with configuration objects, optional seed, and output post-processor.

        :param str output_dir: Directory path where trained weights, metrics, and scaling stats are saved.
        :param EINNModelConfig model_cfg: Neural architecture and compartment/parameter dimension config.
        :param EINNTrainConfig train_cfg: Optimization hyperparameters, phase repetitions, and loss weights.
        :param int | None seed: Global random seed for reproducibility; skips seeding when None (default 42).
        """
        self.output_dir: str = output_dir
        self.model_cfg: EINNModelConfig = model_cfg
        self.train_cfg: EINNTrainConfig = train_cfg
        self.seed: int | None = seed

        if self.seed is not None:
            ExperimentUtils.seed_everything(seed=self.seed)

        # Instantiate post-processor upfront so the output directory exists before training starts.
        self.post_processor = TrainingPostProcessor(output_dir=self.output_dir)

    def train_and_save(
        self,
        dataset: torch.utils.data.Dataset,
        scaling_stats: DataScalingStats,
        model_type: str,
        target_idx: int,
        param_calibration: dict[str, Any],
        epochs: int = 20,
        batch_size: int = 32,
    ):
        """
        Build the single-model EINN, run multiphase training, and serialize artifacts to disk.

        :param torch.utils.data.Dataset dataset: Prepared expanding-window training dataset.
        :param DataScalingStats scaling_stats: Fitted scaling statistics for physical-domain conversion
            and normalized initial-condition supervision.
        :param str model_type: Epidemiological ODE model identifier (e.g. 'SIR', 'SIS', 'SEIRM', 'ROST').
        :param int target_idx: Zero-based index of the target epidemiological compartment in the ODE state vector.
        :param dict[str, Any] param_calibration: Initial physical parameter guesses for ODE initialization.
        :param int epochs: Number of outer training epochs across the multiphase schedule (default 20).
        :param int batch_size: Mini-batch size per gradient step (default 32).
        """
        print(f"Building EINN architecture (ODE Core: {model_type})...")

        # Forward the configured seed so network weight initialization and Fourier sampling are reproducible.
        models = EINNBuilder.build_einn(
            model_config=self.model_cfg,
            train_config=self.train_cfg,
            param_calibration=param_calibration,
            target_idx=target_idx,
            model_type=model_type,
            seed=self.seed,
        )

        # Ensure the ODE model evaluates residuals and supervision against the requested compartment index.
        models.ode_model.target_idx = target_idx

        print("Executing Multiphase Training...")

        trainer = EINNTrainer(
            models=models,
            config=self.train_cfg,
            scaling_stats=scaling_stats,
        )

        metrics = trainer.train(dataset=dataset, epochs=epochs, batch_size=batch_size)

        print("Training complete. Exporting model artifacts...")

        # Persist model state dicts, training metrics, and scaling statistics under the configured prefix.
        self.post_processor.save_models(models=models, prefix="einn")
        self.post_processor.save_metrics(metrics=metrics)
        self.post_processor.save_scaling_stats(scaling_stats=scaling_stats)
