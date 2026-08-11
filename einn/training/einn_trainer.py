import itertools
from typing import List

import torch
from torch.utils.data import DataLoader

from einn.config.einn_config import EINNConfig
from einn.training.einn_forward_engine import EINNForwardEngine
from einn.loss.einn_loss import EINNLoss
from einn.model.interface.einn_models import EINNModels
from einn.model.interface.phase_context import PhaseContext
from einn.training.phase_optimizers.phase_one_optimizer import PhaseOneOptimizer
from einn.training.phase_optimizers.phase_two_optimizer import PhaseTwoOptimizer
from einn.training.phase_optimizers.phase_three_optimizer import PhaseThreeOptimizer
from einn.training.phase_optimizers.phase_four_optimizer import PhaseFourOptimizer
from einn.model.interface.training_metrics import TrainingMetrics


class EINNTrainer:
    """
    Main training orchestrator for the EINN framework.
    Manages optimizers, phase switching, data loading, and metric collection.
    """

    def __init__(self, models: EINNModels, scalers: dict, config: EINNConfig) -> None:
        """
        Initializes the trainer.
        """
        self.models = models
        self.config = config
        self.device = torch.device(config.device)
        self.loss_calculator = EINNLoss(config, models.ode_model, scalers)
        self.forward_engine = EINNForwardEngine(models)

        self.phase_runners = {}
        self.setup_optimizers()

    def setup_optimizers(self) -> None:
        """
        Initializes the PyTorch Adam optimizers for the 4 distinct training phases.
        Uses itertools.chain to cleanly pass parameters from multiple modules to a single optimizer.
        """
        lr = self.config.learning_rate

        # Phase 1 & 2: Feature + Output modules learn
        feat_out_params_1 = itertools.chain(
            self.models.feature_module.parameters(),
            self.models.output_module.parameters()
        )
        opt_1 = torch.optim.Adam(feat_out_params_1, lr=lr)
        self.phase_runners[1] = PhaseOneOptimizer(opt_1, self.forward_engine, self.loss_calculator)

        # We create a fresh Adam instance for Phase 2 to reset momentum states
        feat_out_params_2 = itertools.chain(
            self.models.feature_module.parameters(),
            self.models.output_module.parameters()
        )
        opt_2 = torch.optim.Adam(feat_out_params_2, lr=lr)
        self.phase_runners[2] = PhaseTwoOptimizer(opt_2, self.forward_engine, self.loss_calculator)

        # Phase 3: Time + Output modules learn
        time_out_params = itertools.chain(
            self.models.time_module.parameters(),
            self.models.output_module.parameters()
        )
        opt_3 = torch.optim.Adam(time_out_params, lr=lr)
        self.phase_runners[3] = PhaseThreeOptimizer(opt_3, self.forward_engine, self.loss_calculator)

        # Phase 4: Output module AND ODE parameters learn (Global fine-tuning)
        opt_4_params = itertools.chain(
            self.models.output_module.parameters(),
            self.models.ode_model.parameters()
        )
        # FIX applied: Passed the chained parameters (opt_4_params) to Adam!
        opt_4 = torch.optim.Adam(opt_4_params, lr=lr)
        self.phase_runners[4] = PhaseFourOptimizer(opt_4, self.forward_engine, self.loss_calculator)

    def train(self, dataset, config: EINNConfig) -> List[TrainingMetrics]:
        """
        Executes the main training loop across all phases and epochs.

        Args:
            dataset (EINNDataset): The prepared dataset object.
            config (EINNConfig): Configuration object.

        Returns:
            List[TrainingMetrics]: A list containing the logged metrics for each epoch.
        """
        # Depending on the dataset implementation, batch_size could be dynamic here
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        metrics_list: List[TrainingMetrics] = []

        for phase_num in range(1, 5):
            print(f"--- STARTING PHASE {phase_num} ---")

            if phase_num not in self.phase_runners:
                continue

            phase_optimizer = self.phase_runners[phase_num]

            for epoch in range(config.epochs_per_phase):
                epoch_loss = 0.0

                for batch in dataloader:
                    # Move batch items to the configured device safely
                    X = batch['X'].to(self.device)
                    y = batch['y'].to(self.device)
                    t = batch['t'].to(self.device)
                    aux_targets = batch['aux_targets'].to(self.device)

                    context = PhaseContext(
                        phase_num=phase_num,
                        epoch=epoch,
                        X=X,
                        y=y,
                        t=t,
                        aux_targets=aux_targets,
                        models=self.models
                    )

                    loss = phase_optimizer.step(context)
                    epoch_loss += loss

                # Create a TrainingMetrics record and append it
                metrics = TrainingMetrics(
                    epoch=epoch,
                    phase=phase_num,
                    total_loss=epoch_loss
                )
                metrics_list.append(metrics)

                if epoch % 10 == 0:
                    print(f"Phase {phase_num} | Epoch {epoch} | Total Loss: {epoch_loss:.6f}")

        return metrics_list
