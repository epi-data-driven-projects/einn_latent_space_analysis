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
    Manages the phase optimizers, dataloaders, and coordinates the sequential training epochs.
    """

    def __init__(self, models: EINNModels, config: EINNTrainConfig, scaling_stats: DataScalingStats):
        """
        Initializes the trainer, the forward engine, the loss calculator, and all 4 independent phase optimizers.

        :param EINNModels models: The container holding all initialized neural networks and the ODE model.
        :param EINNTrainConfig config: The training configuration object containing hyperparameters.
        """
        self.models = models
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.engine = EINNForwardEngine(train_config=config, scaling_stats=scaling_stats)
        self.loss_calculator = EINNLoss(config=config, ode_model=models.ode_model, scaling_stats=scaling_stats)

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
              epochs: int, batch_size: int = 1) -> List[TrainingMetrics]:
        """
        Executes the main training loop across all epochs and phases.

        :param EINNDataset dataset: The dataset containing observed inputs, targets, and time vectors.
        :param int epochs: Total number of macroscopic training epochs.
        :param int batch_size: Size of the sliding window batches.
        :return List[TrainingMetrics]: A chronological list of recorded training metrics.
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
            # 1. és 2. FÁZIS EGYBEVONVA (Alternálva)
            # ====================================================
            # Az ismétlésszámot a 'phase_1' kulcsból vesszük.
            reps_p1_2 = self.config.reps.get('phase_1', 300)
            for rep in range(1, reps_p1_2 + 1):
                # Egy iteráció (pass) az adaton a Phase 1-el (Time Solo)
                m1 = self._run_single_rep(self.opt_phase1, phase_num=1, rep=rep, total_reps=reps_p1_2, epoch=epoch, dataloader=dataloader)
                all_metrics.append(m1)
                
                # Azonnal utána egy iteráció a Phase 2-vel (Time + ODE)
                m2 = self._run_single_rep(self.opt_phase2, phase_num=2, rep=rep, total_reps=reps_p1_2, epoch=epoch, dataloader=dataloader)
                all_metrics.append(m2)

            # ====================================================
            # 3. FÁZIS (Tudáslepárlás / Feature Modul)
            # ====================================================
            reps_p3 = self.config.reps.get('phase_3', 300)
            for rep in range(1, reps_p3 + 1):
                m3 = self._run_single_rep(self.opt_phase3, phase_num=3, rep=rep, total_reps=reps_p3, epoch=epoch, dataloader=dataloader)
                all_metrics.append(m3)

            # ====================================================
            # 4. FÁZIS (Gradiens finomhangolás)
            # ====================================================
            reps_p4 = self.config.reps.get('phase_4', 300)
            phase4_metrics = []
            for rep in range(1, reps_p4 + 1):
                m4 = self._run_single_rep(self.opt_phase4, phase_num=4, rep=rep, total_reps=reps_p4, epoch=epoch, dataloader=dataloader)
                phase4_metrics.append(m4)
                all_metrics.append(m4)

            # Early stopping vizsgálat
            # JAVÍTVA: A szótár kulcsa nagy 'F' betűs ('data_F'), nem kicsi!
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
        Executes exactly one repetition (one full pass over the dataloader) for a specific optimizer.
        """
        total_epoch_loss = 0.0
        epoch_loss_components = {}
        batches_count = 0

        for batch_data in dataloader:
            # Construct the PhaseContext using the unpacked batch data
            context = PhaseContext(
                phase_num=phase_num,
                epoch=epoch,
                x=batch_data['x'],
                y=batch_data['y'],
                t=batch_data['t'],
                aux_targets=batch_data['aux_targets'],
                models=self.models,
                mask=batch_data.get('mask', None)
            )

            # Execute the full forward-backward pipeline
            batch_loss, batch_loss_components = phase_optimizer.step(context=context)
            total_epoch_loss += batch_loss

            # Accumulate component losses
            for key, value in batch_loss_components.items():
                epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value

            batches_count += 1

        # Average the loss across all batches
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