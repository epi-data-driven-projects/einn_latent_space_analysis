import logging
from typing import List

import torch

from torch.utils.data import DataLoader

from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.data.einn_dataset import EINNDataset
from einn_multimodel.data.model_balanced_batch_sampler import ModelBalancedBatchSampler
from einn_multimodel.loss.einn_loss import EINNLoss
from einn_multimodel.model.interface.einn_models import EINNModels
from einn_multimodel.model.interface.phase_context import PhaseContext
from einn.model.interface.training_metrics import TrainingMetrics
from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer
from einn.training.early_stopping import EarlyStopping
from einn_multimodel.training.einn_forward_engine import EINNForwardEngine
from einn_multimodel.training.phase_optimizers.phase_1_optimizer import Phase1Optimizer
from einn_multimodel.training.phase_optimizers.phase_2_optimizer import Phase2Optimizer
from einn_multimodel.training.phase_optimizers.phase_3_optimizer import Phase3Optimizer
from einn_multimodel.training.phase_optimizers.phase_4_optimizer import Phase4Optimizer


class EINNTrainer:
    """Four-phase trainer for true-model-only training and SupCon."""

    def __init__(
        self,
        models: EINNModels,
        config: EINNTrainConfig,
        scaling_stats: DataScalingStats,
    ):
        """Initialize EINNTrainer to train each original optimization phase on the provided sequence dataset.


        :param EINNModels models: The models containing the time, feature, output, and ODE modules.
        :param EINNTrainConfig config: Configuration object or mapping controlling this operation.
        :param DataScalingStats scaling_stats: Statistics fitted on training data for physical-domain conversion.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.models = models
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.engine = EINNForwardEngine(config, scaling_stats)
        self.loss_calculator = EINNLoss(config, models, scaling_stats)
        self.opt_phase1 = Phase1Optimizer(models, config, self.engine, self.loss_calculator)
        self.opt_phase2 = Phase2Optimizer(models, config, self.engine, self.loss_calculator)
        self.opt_phase3 = Phase3Optimizer(models, config, self.engine, self.loss_calculator)
        self.opt_phase4 = Phase4Optimizer(models, config, self.engine, self.loss_calculator)

    def _make_dataloader(self, dataset: EINNDataset, batch_size: int) -> DataLoader:
        """Perform make dataloader as part of train each original optimization phase on the provided sequence dataset.


        :param EINNDataset dataset: The prepared padded expanding-window dataset.
        :param int batch_size: Number of samples evaluated in one optimization batch.
        :return DataLoader: Computed result in the configured output representation.
        """
        has_labels = bool(getattr(dataset, "samples", None)) and (
            "true_model_label" in dataset.samples[0]
        )
        if has_labels and len(self.models.ode_models) > 1:
            if self.config.loss_weights.get("supcon", 0.0) > 0:
                sampler = ModelBalancedBatchSampler(dataset, batch_size=batch_size)
                return DataLoader(dataset, batch_sampler=sampler)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(
        self,
        dataset: EINNDataset,
        epochs: int,
        batch_size: int = 1,
    ) -> List[TrainingMetrics]:
        """Run the original multiphase training loop while collecting per-phase loss metrics.


        :param EINNDataset dataset: The prepared padded expanding-window dataset.
        :param int epochs: Number of complete training passes requested.
        :param int batch_size: Number of samples evaluated in one optimization batch.
        :return List[TrainingMetrics]: Collection of the generated results in their original order.
        """
        if bool(getattr(dataset, "samples", None)):
            for sample in dataset.samples:
                true_label = sample.get("true_model_label")
                assumed_label = sample.get("assumed_model_label")
                if true_label is None:
                    raise ValueError("Every training sample needs true_model_label.")
                if assumed_label is not None and not bool(torch.equal(
                    torch.as_tensor(true_label), torch.as_tensor(assumed_label)
                )):
                    raise ValueError(
                        "True-only training received a mismatched sample. "
                        "Generate data with pair_with_all_assumptions=False."
                    )
        if self.config.mask_mismatched_losses:
            raise ValueError("Disable mask_mismatched_losses for true-only training.")
        dataloader = self._make_dataloader(dataset, batch_size)
        all_metrics: List[TrainingMetrics] = []
        early_stopping = EarlyStopping(
            mode="min",
            min_delta=self.config.early_stopping_min_delta,
            patience=self.config.early_stopping_patience,
            percentage=self.config.early_stopping_percentage,
        )

        self.logger.info("Starting multi-model EINN training loop.")
        for epoch in range(1, epochs + 1):
            reps_p1_2 = self.config.reps.get("phase_1", 300)
            for rep in range(1, reps_p1_2 + 1):
                all_metrics.append(self._run_single_rep(
                    self.opt_phase1, 1, rep, reps_p1_2, epoch, dataloader
                ))
                all_metrics.append(self._run_single_rep(
                    self.opt_phase2, 2, rep, reps_p1_2, epoch, dataloader
                ))

            reps_p3 = self.config.reps.get("phase_3", 300)
            for rep in range(1, reps_p3 + 1):
                all_metrics.append(self._run_single_rep(
                    self.opt_phase3, 3, rep, reps_p3, epoch, dataloader
                ))

            reps_p4 = self.config.reps.get("phase_4", 300)
            phase4_metrics = []
            for rep in range(1, reps_p4 + 1):
                metric = self._run_single_rep(
                    self.opt_phase4, 4, rep, reps_p4, epoch, dataloader
                )
                phase4_metrics.append(metric)
                all_metrics.append(metric)

            if phase4_metrics and "data_F" in phase4_metrics[-1].loss_components:
                monitored_loss = phase4_metrics[-1].loss_components["data_F"]
            elif all_metrics:
                monitored_loss = all_metrics[-1].total_loss
            else:
                continue
            if early_stopping.step(metrics=monitored_loss):
                self.logger.info("Early stopping triggered at epoch %s.", epoch)
                break
        return all_metrics

    def _run_single_rep(
        self,
        phase_optimizer: BasePhaseOptimizer,
        phase_num: int,
        rep: int,
        total_reps: int,
        epoch: int,
        dataloader: DataLoader,
    ) -> TrainingMetrics:
        """Perform run single rep as part of train each original optimization phase on the provided sequence dataset.


        :param BasePhaseOptimizer phase_optimizer: Optimization helper responsible for the active training phase.
        :param int phase_num: Number of the currently active training phase.
        :param int rep: Repetition index within the selected optimization phase.
        :param int total_reps: Total number of repetitions configured for the phase.
        :param int epoch: Index of the current outer training epoch.
        :param DataLoader dataloader: Iterator that yields supervised and masked trajectory batches.
        :return TrainingMetrics: Computed result in the configured output representation.
        """
        total_loss = 0.0
        component_sums: dict[str, float] = {}
        batches = 0

        for batch_data in dataloader:
            context = PhaseContext(
                phase_num=phase_num,
                epoch=epoch,
                x=batch_data["X"],
                y=batch_data["y"],
                t=batch_data["t"],
                aux_targets=batch_data["aux_targets"],
                models=self.models,
                mask=batch_data.get("mask"),
                true_model_label=batch_data.get("true_model_label"),
                # Compatibility alias for the loss interface; ALWAYS true label.
                assumed_model_label=batch_data.get("true_model_label"),
                is_mismatch=None,
            )
            batch_loss, batch_components = phase_optimizer.step(context)
            total_loss += batch_loss
            for key, value in batch_components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value
            batches += 1

        average_loss = total_loss / max(batches, 1)
        average_components = {
            key: value / max(batches, 1)
            for key, value in component_sums.items()
        }
        metric = TrainingMetrics(
            epoch=epoch,
            phase=phase_num,
            rep=rep,
            total_loss=average_loss,
            loss_components=average_components,
        )
        print(
            f"Epoch: {epoch} | Phase: {phase_num} | Rep: {rep}/{total_reps} "
            f"| Total Loss: {metric.total_loss:.4e}"
        )
        return metric
