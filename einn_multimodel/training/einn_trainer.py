"""
Four-phase training orchestrator for true-model-only multi-model EINN with SupCon.

Coordinates Phase1Optimizer through Phase4Optimizer, model-balanced batch sampling when
SupConLoss is active, strict true-only dataset/config validation, early stopping on Phase 4
Feature data loss, and per-repetition TrainingMetrics logging (thesis Sections 2.2-2.4).
"""
import logging
from typing import List

import torch
from torch.utils.data import DataLoader

from einn.data.data_scaling_stats import DataScalingStats
from einn.data.einn_dataset import EINNDataset
from einn.model.interface.training_metrics import TrainingMetrics
from einn.training.early_stopping import EarlyStopping
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.data.model_balanced_batch_sampler import ModelBalancedBatchSampler
from einn_multimodel.loss.einn_loss import EINNLoss
from einn_multimodel.model.interface.einn_models import EINNModels
from einn_multimodel.model.interface.phase_context import PhaseContext
from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer
from einn_multimodel.training.einn_forward_engine import EINNForwardEngine
from einn_multimodel.training.phase_optimizers.phase_1_optimizer import Phase1Optimizer
from einn_multimodel.training.phase_optimizers.phase_2_optimizer import Phase2Optimizer
from einn_multimodel.training.phase_optimizers.phase_3_optimizer import Phase3Optimizer
from einn_multimodel.training.phase_optimizers.phase_4_optimizer import Phase4Optimizer


class EINNTrainer:
    """
    Four-phase trainer for multi-model EINN runs where assumed labels match true model labels.

    Constructs the shared EINNForwardEngine, EINNLoss, and four phase-specific Adam optimizers,
    alternates Phase 1 and Phase 2 repetitions inside each epoch, runs Phase 3 (with
    Supervised Contrastive loss on Feature embeddings) and Phase 4 (Feature ODE fine-tuning
    via the gradient trick), and checks early stopping after each epoch (thesis Sections 2.2-2.4).
    """

    def __init__(
        self,
        models: EINNModels,
        config: EINNTrainConfig,
        scaling_stats: DataScalingStats,
    ):
        """
        Initialize the trainer, forward engine, composite loss, and four phase optimizers.

        :param EINNModels models: Multi-model EINN container holding time_modules, ode_models,
            feature_module, output_module, and model_to_id.
        :param EINNTrainConfig config: Training configuration specifying phase repetitions,
            learning rate, future_steps, loss_weights, and early-stopping parameters.
        :param DataScalingStats scaling_stats: Training-fitted scaling metadata for physical
            unit conversions and normalized initial-state penalties.
        """
        self.models: EINNModels = models
        self.config: EINNTrainConfig = config
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.engine: EINNForwardEngine = EINNForwardEngine(config, scaling_stats)
        self.loss_calculator: EINNLoss = EINNLoss(config, models, scaling_stats)
        self.opt_phase1: Phase1Optimizer = Phase1Optimizer(models, config, self.engine, self.loss_calculator)
        self.opt_phase2: Phase2Optimizer = Phase2Optimizer(models, config, self.engine, self.loss_calculator)
        self.opt_phase3: Phase3Optimizer = Phase3Optimizer(models, config, self.engine, self.loss_calculator)
        self.opt_phase4: Phase4Optimizer = Phase4Optimizer(models, config, self.engine, self.loss_calculator)

    def _make_dataloader(self, dataset: EINNDataset, batch_size: int) -> DataLoader:
        """
        Build a DataLoader using ModelBalancedBatchSampler when multi-model SupCon is active.

        Balanced sampling across model classes in each batch ensures every anchor in SupConLoss
        has positive and negative pairs across the paired epidemiological models (thesis Section 2.4).

        :param EINNDataset dataset: Expanding-window sequence dataset with optional model labels.
        :param int batch_size: Target mini-batch size per iteration.
        :return DataLoader: Configured PyTorch DataLoader over dataset.
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
        """
        Execute the four-phase EINN training loop across epochs and return collected metrics.

        Validates that samples do not contain mismatched true/assumed model labels and that
        mask_mismatched_losses is disabled, then runs alternating Phase 1 / Phase 2 repetitions,
        Phase 3 repetitions, and Phase 4 repetitions per epoch with early stopping.

        :param EINNDataset dataset: Training dataset of expanding-window trajectories.
        :param int epochs: Maximum number of outer training epochs.
        :param int batch_size: Mini-batch size used when shuffle=True, or target batch size
            passed to ModelBalancedBatchSampler.

        :return List[TrainingMetrics]: Chronological list of per-repetition TrainingMetrics entries.

        :raises ValueError: If a sample lacks true_model_label, contains mismatched
            true/assumed labels, or if config.mask_mismatched_losses is True.
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
            # Alternate Phase 1 (Time + Output) and Phase 2 (Time + ODE + Output) per repetition.
            reps_p1_2 = self.config.reps.get("phase_1", 300)
            for rep in range(1, reps_p1_2 + 1):
                all_metrics.append(self._run_single_rep(
                    self.opt_phase1, 1, rep, reps_p1_2, epoch, dataloader
                ))
                all_metrics.append(self._run_single_rep(
                    self.opt_phase2, 2, rep, reps_p1_2, epoch, dataloader
                ))

            # Phase 3: train Feature, Output, and Time modules with SupCon and distillation.
            reps_p3 = self.config.reps.get("phase_3", 300)
            for rep in range(1, reps_p3 + 1):
                all_metrics.append(self._run_single_rep(
                    self.opt_phase3, 3, rep, reps_p3, epoch, dataloader
                ))

            # Phase 4: fine-tune ODE parameters and OutputModule on Feature ODE residuals.
            reps_p4 = self.config.reps.get("phase_4", 300)
            phase4_metrics = []
            for rep in range(1, reps_p4 + 1):
                metric = self._run_single_rep(
                    self.opt_phase4, 4, rep, reps_p4, epoch, dataloader
                )
                phase4_metrics.append(metric)
                all_metrics.append(metric)

            # Monitor Phase 4 Feature data loss when available, falling back to last total loss.
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
        """
        Run one full pass over dataloader for a single repetition of a training phase.

        :param BasePhaseOptimizer phase_optimizer: Phase optimizer executing step() on each batch.
        :param int phase_num: Active phase identifier in {1, 2, 3, 4}.
        :param int rep: 1-based repetition index within the current phase.
        :param int total_reps: Total repetitions scheduled for this phase per epoch.
        :param int epoch: 1-based outer training epoch index.
        :param DataLoader dataloader: DataLoader yielding trajectory batch dictionaries.

        :return TrainingMetrics: Epoch/phase/rep summary containing average total and component losses.
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
                # In true-only training, assumed_model_label is set equal to true_model_label.
                assumed_model_label=batch_data.get("true_model_label"),
                is_mismatch=None,
            )
            batch_loss, batch_components = phase_optimizer.step(context)
            total_loss += batch_loss
            for key, value in batch_components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value
            batches += 1

        # Average batch losses across all iterations completed in this repetition.
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
