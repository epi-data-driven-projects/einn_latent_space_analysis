import logging
from typing import List, Dict

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
    Orchestrates the multi-phase training process of the Epidemiologically-Informed Neural Network (EINN).
    Responsible for initializing the execution engine, loss calculation, and phase-specific optimizers,
    and managing the chronological execution of training epochs and phases.
    """

    def __init__(self, models: EINNModels, config: EINNTrainConfig):
        """
        Initializes the trainer along with its dependency tree (Engine, Loss, Optimizers).

        :param EINNModels models: Dataclass containing the initialized neural and ODE models.
        :param EINNTrainConfig config: Configuration object containing learning rates, weights, and device info.
        """
        self.models = models
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize the shared execution engine and loss calculator
        self.engine = EINNForwardEngine(train_config=config)
        self.loss_calculator = EINNLoss(config=config, ode_model=models.ode_model)

        # Initialize the 4 separate phase optimizers.
        # This design pattern ensures each optimizer maintains its own distinct Adam momentum state
        # across different epochs, precisely mirroring the original EINN implementation.
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
        :param int batch_size: Size of the sliding window batches. Defaults to 1.
        :return List[TrainingMetrics]: A chronological list of fully detailed training metrics for analysis.
        """
        all_metrics: List[TrainingMetrics] = []

        # Instantiate the PyTorch DataLoader to handle batching and shuffling
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.logger.info("Starting EINN Training Loop...")

        for epoch in range(1, epochs + 1):
            self.logger.info(f"--- EPOCH {epoch}/{epochs} ---")

            # Sequentially execute the 4 phases within the same epoch.
            # extend() appends the generated lists of TrainingMetrics into a flat chronological list.
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
        Executes a specific phase for a given number of repetitions over the entire dataset.
        Aggregates batch losses into epoch-level metrics.

        :param BasePhaseOptimizer optimizer: The specific optimizer instance for this phase.
        :param int phase_num: The identifier of the phase (1, 2, 3, or 4).
        :param int reps: How many times to loop over the dataloader in this phase.
        :param int epoch: The current macroscopic epoch number.
        :param DataLoader dataloader: The data provider instance.
        :return List[TrainingMetrics]: A list of metrics recorded during this phase's execution.
        """
        metrics_list = []

        # A phase can be repeated multiple times within a single macroscopic epoch
        for rep in range(1, reps + 1):

            # Dictionary to accumulate all individual loss components over the batches
            epoch_losses: Dict[str, float] = {
                'total_loss': 0.0, 'loss_data_T': 0.0, 'loss_data_F': 0.0, 'loss_aux': 0.0,
                'loss_ode_T': 0.0, 'loss_ode_F': 0.0, 'loss_ode_future_T': 0.0, 'loss_ode_future_F': 0.0,
                'loss_mono': 0.0, 'loss_param': 0.0, 'loss_kd_target': 0.0, 'loss_kd_emb': 0.0
            }
            batches_count = 0

            for batch_data in dataloader:
                # Build the context object required by the Forward Engine and Optimizers
                context = PhaseContext(
                    phase_num=phase_num,
                    epoch=epoch,
                    x=batch_data['x'],
                    y=batch_data['y'],
                    t=batch_data['t'],
                    aux_targets=batch_data['aux_targets'],
                    models=self.models
                )

                # Execute the Forward -> Loss -> Backward -> Clip -> Step pipeline
                batch_losses = optimizer.step(context=context)

                # Accumulate the returned partial losses
                for key, val in batch_losses.items():
                    if key in epoch_losses:
                        epoch_losses[key] += val

                batches_count += 1

            # Average the accumulated losses by the number of batches
            avg_losses = {k: v / max(1, batches_count) for k, v in epoch_losses.items()}

            # Populate the metrics dataclass
            metrics = TrainingMetrics(
                epoch=epoch, phase=phase_num, rep=rep,
                total_loss=avg_losses['total_loss'],
                loss_data_T=avg_losses['loss_data_T'], loss_data_F=avg_losses['loss_data_F'],
                loss_aux=avg_losses['loss_aux'], loss_ode_T=avg_losses['loss_ode_T'],
                loss_ode_F=avg_losses['loss_ode_F'], loss_ode_future_T=avg_losses['loss_ode_future_T'],
                loss_ode_future_F=avg_losses['loss_ode_future_F'], loss_mono=avg_losses['loss_mono'],
                loss_param=avg_losses['loss_param'], loss_kd_target=avg_losses['loss_kd_target'],
                loss_kd_emb=avg_losses['loss_kd_emb']
            )
            metrics_list.append(metrics)

            # Optional console feedback
            print(f"Epoch: {epoch} | Phase: {phase_num} | Rep: {rep}/{reps} | Total Loss: {metrics.total_loss:.4e}")

        return metrics_list
