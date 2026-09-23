import logging
from typing import List

from torch.utils.data import DataLoader

from einn.config.einn_train_config import EINNTrainConfig
from einn.data.einn_dataset import EINNDataset
from einn.loss.einn_loss import EINNLoss
from einn.model.interface.einn_models import EINNModels
from einn.model.interface.phase_context import PhaseContext
from einn.model.interface.training_metrics import TrainingMetrics
from einn.training.einn_forward_engine import EINNForwardEngine
from einn.training.phase_optimizers.phase_1_optimizer import Phase1Optimizer
from einn.training.phase_optimizers.phase_2_optimizer import Phase2Optimizer
from einn.training.phase_optimizers.phase_3_optimizer import Phase3Optimizer
from einn.training.phase_optimizers.phase_4_optimizer import Phase4Optimizer


class EINNTrainer:
    """
    Orchestrates the multiphase training process of the Epidemiologically-Informed Neural Network (EINN).
    Manages the phase optimizers, dataloaders, and coordinates the sequential training epochs.
    """
    def __init__(self, models: EINNModels, config: EINNTrainConfig):
        """
        Initializes the trainer, the forward engine, the loss calculator, and all 4 independent phase optimizers.

        :param EINNModels models: The container holding all initialized neural networks and the ODE model.
        :param EINNTrainConfig config: The training configuration object containing hyperparameters.
        """
        self.models = models
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.engine = EINNForwardEngine(train_config=config)
        self.loss_calculator = EINNLoss(config=config, ode_model=models.ode_model)

        # Initialize the 4 separate Phase Optimizers, preserving their independent Adam states
        self.opt_phase1 = Phase1Optimizer(models=models, config=config, engine=self.engine,
                                          loss_calculator=self.loss_calculator)
        self.opt_phase2 = Phase2Optimizer(models=models, config=config, engine=self.engine,
                                          loss_calculator=self.loss_calculator)
        self.opt_phase3 = Phase3Optimizer(models=models, config=config, engine=self.engine,
                                          loss_calculator=self.loss_calculator)
        self.opt_phase4 = Phase4Optimizer(models=models, config=config, engine=self.engine,
                                          loss_calculator=self.loss_calculator)

    def train(self, dataset: EINNDataset,
               epochs: int, reps: dict[str, int], batch_size: int = 1) -> List[TrainingMetrics]:
        """
        Executes the main training loop across all epochs and phases.

        :param EINNDataset dataset: The dataset containing observed inputs, targets, and time vectors.
        :param int epochs: Total number of macroscopic training epochs.
        :param dict[str, int] reps: Repetition counts for each phase. Expected keys: 'phase_1', 'phase_2', etc.
        :param int batch_size: Size of the sliding window batches.
        :return List[TrainingMetrics]: A chronological list of recorded training metrics.
        """
        all_metrics: List[TrainingMetrics] = []
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.logger.info("Starting EINN Training Loop...")

        for epoch in range(1, epochs + 1):
            self.logger.info(f"--- EPOCH {epoch}/{epochs} ---")

            # Execute phases sequentially based on the provided repetition dictionary
            all_metrics.extend(self._run_phase(
                optimizer=self.opt_phase1, phase_num=1, reps=reps['phase_1'], epoch=epoch, dataloader=dataloader))

            all_metrics.extend(self._run_phase(
                optimizer=self.opt_phase2, phase_num=2, reps=reps['phase_2'], epoch=epoch, dataloader=dataloader))

            all_metrics.extend(self._run_phase(
                optimizer=self.opt_phase3, phase_num=3, reps=reps['phase_3'], epoch=epoch, dataloader=dataloader))

            all_metrics.extend(self._run_phase(
                optimizer=self.opt_phase4, phase_num=4, reps=reps['phase_4'], epoch=epoch, dataloader=dataloader))

        return all_metrics

    def _run_phase(self, optimizer,
                   phase_num: int, reps: int, epoch: int, dataloader: DataLoader) -> List[TrainingMetrics]:
        """
        Executes a specific optimization phase for a given number of repetitions over the entire dataset.

        :param BasePhaseOptimizer optimizer: The specific optimizer instance handling the current phase.
        :param int phase_num: The numerical identifier of the current phase (1, 2, 3, or 4).
        :param int reps: The number of times to iterate over the entire dataloader in this phase.
        :param int epoch: The current macroscopic epoch number.
        :param DataLoader dataloader: The PyTorch dataloader providing batched sequences.
        :return List[TrainingMetrics]: Metrics recorded during this phase's execution.
        """
        metrics_list = []

        for rep in range(1, reps + 1):
            total_epoch_loss = 0.0
            batches_count = 0

            for batch_data in dataloader:
                # Construct the PhaseContext using the unpacked batch data
                context = PhaseContext(
                    phase_num=phase_num,
                    epoch=epoch,
                    x=batch_data['X'],
                    y=batch_data['y'],
                    t=batch_data['t'],
                    aux_targets=batch_data['aux_targets'],
                    models=self.models
                )

                # Execute the full forward-backward pipeline and retrieve the scalar total loss
                batch_loss = optimizer.step(context=context)

                total_epoch_loss += batch_loss
                batches_count += 1

            # Average the loss across all batches processed
            avg_loss = total_epoch_loss / max(1, batches_count)

            metrics = TrainingMetrics(
                epoch=epoch, phase=phase_num, rep=rep,
                total_loss=avg_loss
            )
            metrics_list.append(metrics)

            print(f"Epoch: {epoch} | Phase: {phase_num} | Rep: {rep}/{reps} | Total Loss: {metrics.total_loss:.4e}")

        return metrics_list
