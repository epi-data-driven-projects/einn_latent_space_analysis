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
    # TODO: add docstring, comments
    """
    Lorem ipsum
    """
    def __init__(self, models: EINNModels, config: EINNTrainConfig):
        """
        Asd
        :param models:
        :param config:
        """
        self.models = models
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.engine = EINNForwardEngine(train_config=config)
        self.loss_calculator = EINNLoss(config=config, ode_model=models.ode_model)

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
        :param dict[str, int] reps: Repetition counts for each phase.
        :param int batch_size: Size of the sliding window batches.
        :return List[TrainingMetrics]: A chronological list of full training metrics.
        """
        all_metrics: List[TrainingMetrics] = []
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.logger.info("Starting EINN Training Loop...")

        for epoch in range(1, epochs + 1):
            self.logger.info(f"--- EPOCH {epoch}/{epochs} ---")

            if reps.get('time', 0) > 0:
                all_metrics.extend(self._run_phase(
                    optimizer=self.opt_phase1, phase_num=1, reps=reps['time'], epoch=epoch, dataloader=dataloader))

            if reps.get('time_ode', 0) > 0:
                all_metrics.extend(self._run_phase(
                    optimizer=self.opt_phase2, phase_num=2, reps=reps['time_ode'], epoch=epoch, dataloader=dataloader))

            if reps.get('feat_time', 0) > 0:
                all_metrics.extend(self._run_phase(
                    optimizer=self.opt_phase3, phase_num=3, reps=reps['feat_time'], epoch=epoch, dataloader=dataloader))

            if reps.get('out', 0) > 0:
                all_metrics.extend(self._run_phase(
                    optimizer=self.opt_phase4, phase_num=4, reps=reps['out'], epoch=epoch, dataloader=dataloader))

        return all_metrics

    def _run_phase(self, optimizer,
                   phase_num: int, reps: int, epoch: int, dataloader: DataLoader) -> List[TrainingMetrics]:
        metrics_list = []

        for rep in range(1, reps + 1):
            epoch_losses: Dict[str, float] = {
                'total_loss': 0.0, 'loss_data_T': 0.0, 'loss_data_F': 0.0, 'loss_aux': 0.0,
                'loss_ode_T': 0.0, 'loss_ode_F': 0.0, 'loss_ode_future_T': 0.0, 'loss_ode_future_F': 0.0,
                'loss_mono': 0.0, 'loss_param': 0.0, 'loss_kd_target': 0.0, 'loss_kd_emb': 0.0
            }
            batches_count = 0

            for batch_data in dataloader:
                context = PhaseContext(
                    phase_num=phase_num,
                    epoch=epoch,
                    x=batch_data['x'],
                    y=batch_data['y'],
                    t=batch_data['t'],
                    aux_targets=batch_data['aux_targets'],
                    models=self.models
                )

                batch_losses = optimizer.step(context=context)

                for key, val in batch_losses.items():
                    if key in epoch_losses:
                        epoch_losses[key] += val

                batches_count += 1

            avg_losses = {k: v / max(1, batches_count) for k, v in epoch_losses.items()}

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

            print(f"Epoch: {epoch} | Phase: {phase_num} | Rep: {rep}/{reps} | Total Loss: {metrics.total_loss:.4e}")

        return metrics_list
