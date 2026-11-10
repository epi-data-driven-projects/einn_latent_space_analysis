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
    Lightweight orchestration class for the EINN framework.

    This class is responsible strictly for coordinating the training process:
    it enforces experiment reproducibility, configures the neural networks,
    builds the ODE core, executes the multiphase training loop, and delegates
    the serialization of the resulting model weights and metrics to the PostProcessor.
    """

    def __init__(
            self,
            output_dir: str,
            model_cfg: EINNModelConfig,
            train_cfg: EINNTrainConfig,
            seed: int = 42
    ):
        """
        Initializes the runner with pure configuration objects and sets the global seed.

        :param str output_dir: The directory path where all training artifacts will be saved.
        :param EINNModelConfig model_cfg: Configuration detailing the architectural dimensions.
        :param EINNTrainConfig train_cfg: Configuration detailing the training hyperparameters.
        :param int seed: The random seed to ensure global determinism (default: 42).
        """
        self.output_dir = output_dir
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.seed = seed

        if self.seed is not None:
            ExperimentUtils.seed_everything(seed=self.seed)

        # The post-processor is instantiated immediately to ensure the output directory is ready.
        self.post_processor = TrainingPostProcessor(output_dir=self.output_dir)

    def train_and_save(
            self,
            dataset: torch.utils.data.Dataset,
            scaling_stats: DataScalingStats,
            model_type: str,
            target_idx: int,
            param_calibration: dict,
            epochs: int = 20,
            batch_size: int = 32
    ):
        """
        Constructs the EINN architecture, executes the trainer, and saves the artifacts to disk.

        :param torch.utils.data.Dataset dataset: The prepared expanding-window PyTorch Dataset.
        :param DataScalingStats scaling_stats: The runtime scaling statistics (means, standard deviations, and boundaries).
        :param str model_type: The physical ODE core to instantiate (e.g., "SEIRM", "SIR").
        :param int target_idx: The integer index of the targeted epidemiological compartment.
        :param dict param_calibration: The physical initial guesses for the ODE parameters.
        :param int epochs: Total number of complete passes through the dataset per training phase.
        :param int batch_size: Number of samples per gradient update.
        """
        print(f"Building EINN architecture (ODE Core: {model_type})...")

        # Az EINNBuilder most már dinamikusan megkapja a Runnerhez rendelt seed-et
        models = EINNBuilder.build_einn(
            model_config=self.model_cfg,
            train_config=self.train_cfg,
            param_calibration=param_calibration,
            model_type=model_type,
            seed=self.seed
        )

        # Explicitly inform the underlying mathematical ODE model which compartment to target for the loss calculations.
        models.ode_model.target_idx = target_idx

        print("Executing Multiphase Training...")

        # Instantiate the Trainer
        trainer = EINNTrainer(
            models=models,
            config=self.train_cfg,
            scaling_stats=scaling_stats
        )

        # Execute the training loop and collect the progression metrics.
        metrics = trainer.train(dataset=dataset, epochs=epochs, batch_size=batch_size)

        print("Training complete. Exporting model artifacts...")

        # Serialize all final outputs using the PostProcessor.
        self.post_processor.save_models(models=models, prefix="einn")
        self.post_processor.save_metrics(metrics=metrics)
        self.post_processor.save_scaling_stats(scaling_stats=scaling_stats)
