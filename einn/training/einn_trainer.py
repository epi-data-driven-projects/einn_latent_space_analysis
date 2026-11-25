import logging
from typing import List

from torch.utils.data import DataLoader

from einn.config.einn_train_config import EINNTrainConfig
from einn.data.einn_dataset import EINNDataset
from einn.loss.einn_loss import EINNLoss
from einn.model.interface.einn_models import EINNModels
from einn.model.interface.phase_context import PhaseContext
from einn.model.interface.training_metrics import TrainingMetrics
from einn.training.base_phase_optimizer import BasePhaseOptimizer
from einn.data.data_scaling_stats import DataScalingStats
from einn.training.early_stopping import EarlyStopping
from einn.training.einn_forward_engine import EINNForwardEngine
from einn.training.phase_optimizers.phase_1_optimizer import Phase1Optimizer
from einn.training.phase_optimizers.phase_2_optimizer import Phase2Optimizer
from einn.training.phase_optimizers.phase_3_optimizer import Phase3Optimizer
from einn.training.phase_optimizers.phase_4_optimizer import Phase4Optimizer


class EINNTrainer:
    """
    Orchestrates the multiphase training process of the Epidemiologically-Informed Neural Network (EINN).

    This trainer manages the complex execution flow of the four distinct training phases,
    coordinates data loading, applies early stopping logic, and handles the persistence
    of individual phase optimizers to ensure momentum and Adam states are maintained
    across training iterations.
    """

    def __init__(self, models: EINNModels, config: EINNTrainConfig, scaling_stats: DataScalingStats):
        """
        Initializes the trainer, the forward computation engine, the loss calculator,
        and all four independent phase optimizers.

        :param EINNModels models: The container holding all initialized neural networks
            (Time Module, Feature Module, Output Module) and the underlying physical ODE model.
        :param EINNTrainConfig config: The training configuration object containing hyperparameters,
            learning rates, loss weights, and phase repetition counts.
        :param DataScalingStats scaling_stats: The global statistics (mean, std, time bounds)
            used to reverse-scale predictions during evaluation and physical loss calculations.
        """
        self.models = models
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.engine = EINNForwardEngine(train_config=config, scaling_stats=scaling_stats)
        self.loss_calculator = EINNLoss(config=config, ode_model=models.ode_model, scaling_stats=scaling_stats)

        # Initialize the 4 separate Phase Optimizers, preserving their independent Adam states
        # Each phase activates/freezes different parts of the network architecture.
        self.opt_phase1 = Phase1Optimizer(models=models, config=config, engine=self.engine,
                                          loss_calculator=self.loss_calculator)
        self.opt_phase2 = Phase2Optimizer(models=models, config=config, engine=self.engine,
                                          loss_calculator=self.loss_calculator)
        self.opt_phase3 = Phase3Optimizer(models=models, config=config, engine=self.engine,
                                          loss_calculator=self.loss_calculator)
        self.opt_phase4 = Phase4Optimizer(models=models, config=config, engine=self.engine,
                                          loss_calculator=self.loss_calculator)

    def train(self, dataset: EINNDataset,
              epochs: int, batch_size: int = 1) -> List[TrainingMetrics]:
        """
        Executes the main training loop across all macroscopic epochs and internal phases.

        The EINN training process is highly structured. Within each macroscopic epoch,
        the network cycles through Phase 1 and 2 in an alternating fashion, followed by
        Phase 3 (Knowledge Distillation) and Phase 4 (Gradient fine-tuning).

        :param EINNDataset dataset: The dataset containing observed inputs, targets,
            covariates, physical masks, and time vectors.
        :param int epochs: Total number of macroscopic training epochs to execute.
        :param int batch_size: Size of the sliding window batches for stochastic gradient descent.
            Defaults to 1 for precise trajectory fitting.
        :return List[TrainingMetrics]: A chronological list of recorded training metrics detailing
            the loss components for every repetition of every phase.
        """
        all_metrics: List[TrainingMetrics] = []
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        early_stopping = EarlyStopping(
            mode='min',
            min_delta=self.config.early_stopping_min_delta,
            patience=self.config.early_stopping_patience,
            percentage=self.config.early_stopping_percentage
        )

        self.logger.info("Starting EINN Training Loop...")

        for epoch in range(1, epochs + 1):
            self.logger.info(f"--- EPOCH {epoch}/{epochs} ---")

            # ====================================================
            # PHASE 1 & 2 INTERLEAVED (Time Solo + Physical Constraints)
            # ====================================================
            # To ensure the Time Module learns the data while respecting physical boundaries,
            # Phase 1 (Data fit) and Phase 2 (ODE constraints) are executed in an alternating manner.
            reps_p1_2 = self.config.reps.get('phase_1', 300)
            for rep in range(1, reps_p1_2 + 1):
                # Execute one full pass over the data with Phase 1 (Data-only focus)
                m1 = self._run_single_rep(self.opt_phase1, phase_num=1, rep=rep, total_reps=reps_p1_2, epoch=epoch,
                                          dataloader=dataloader)
                all_metrics.append(m1)

                # Immediately follow up with one pass of Phase 2 (Physics-informed focus)
                m2 = self._run_single_rep(self.opt_phase2, phase_num=2, rep=rep, total_reps=reps_p1_2, epoch=epoch,
                                          dataloader=dataloader)
                all_metrics.append(m2)

            # ====================================================
            # PHASE 3 (Knowledge Distillation)
            # ====================================================
            # Unfreezes the Feature Module. The 'Student' network learns from the raw data
            # while being heavily guided by the 'Teacher' (Time Module) representations.
            reps_p3 = self.config.reps.get('phase_3', 300)
            for rep in range(1, reps_p3 + 1):
                m3 = self._run_single_rep(self.opt_phase3, phase_num=3, rep=rep, total_reps=reps_p3, epoch=epoch,
                                          dataloader=dataloader)
                all_metrics.append(m3)

            # ====================================================
            # PHASE 4 (Gradient Fine-tuning)
            # ====================================================
            # Freezes both feature extractors and fine-tunes only the shared Output Module
            # and the physical ODE parameters using the chain-rule derived gradients.
            reps_p4 = self.config.reps.get('phase_4', 300)
            phase4_metrics = []
            for rep in range(1, reps_p4 + 1):
                m4 = self._run_single_rep(self.opt_phase4, phase_num=4, rep=rep, total_reps=reps_p4, epoch=epoch,
                                          dataloader=dataloader)
                phase4_metrics.append(m4)
                all_metrics.append(m4)

            # Check Early Stopping criteria based on the Feature Module's data loss
            if len(phase4_metrics) > 0 and 'data_F' in phase4_metrics[-1].loss_components:
                monitored_loss = phase4_metrics[-1].loss_components['data_F']
            elif len(all_metrics) > 0:
                monitored_loss = all_metrics[-1].total_loss
            else:
                continue

            should_stop = early_stopping.step(metrics=monitored_loss)
            if should_stop:
                self.logger.info(
                    f"Early Stopping triggered at epoch {epoch}! No improvement for {early_stopping.patience} epochs.")
                break

        return all_metrics

    def _run_single_rep(self, phase_optimizer: BasePhaseOptimizer, phase_num: int,
                        rep: int, total_reps: int, epoch: int, dataloader: DataLoader) -> TrainingMetrics:
        """
        Executes exactly one repetition (one full pass over the dataloader) for a specific phase optimizer.

        This method handles the extraction of batch data, constructing the computational context,
        triggering the forward/backward passes, and averaging the loss metrics across all batches.

        :param BasePhaseOptimizer phase_optimizer: The specific optimizer handling the current phase logic.
        :param int phase_num: The integer ID (1-4) of the current training phase.
        :param int rep: The current repetition index within the phase.
        :param int total_reps: The total number of planned repetitions for this phase (used for logging).
        :param int epoch: The current macroscopic epoch index.
        :param DataLoader dataloader: The PyTorch DataLoader providing the batched sequence data.
        :return TrainingMetrics: An object containing the averaged total loss and individual loss components.
        """
        total_epoch_loss = 0.0
        epoch_loss_components = {}
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
                models=self.models,
                mask=batch_data.get('mask', None)
            )

            # Execute the full forward-backward pipeline for this specific batch
            batch_loss, batch_loss_components = phase_optimizer.step(context=context)
            total_epoch_loss += batch_loss

            # Accumulate internal component losses (e.g., data_T, ode_F, mono)
            for key, value in batch_loss_components.items():
                epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value

            batches_count += 1

        # Average the accumulated losses across all processed batches
        avg_loss = total_epoch_loss / max(1, batches_count)
        avg_components = {k: v / max(1, batches_count) for k, v in epoch_loss_components.items()}

        metrics = TrainingMetrics(
            epoch=epoch,
            phase=phase_num,
            rep=rep,
            total_loss=avg_loss,
            loss_components=avg_components
        )
        print(f"Epoch: {epoch} | Phase: {phase_num} | Rep: {rep}/{total_reps} | Total Loss: {metrics.total_loss:.4e}")

        return metrics
