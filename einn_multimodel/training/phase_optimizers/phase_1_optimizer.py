import itertools

import torch

from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer


class Phase1Optimizer(BasePhaseOptimizer):
    """Phase 1: train all Time Modules and the shared Output Module."""

    def _init_optimizer(self) -> None:
        params = itertools.chain(
            self.models.time_modules.parameters(),
            self.models.output_module.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.learning_rate, amsgrad=True
        )

    def prepare_network_states(self) -> None:
        self._set_trainable(self.models.time_modules, True)
        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_models, False)
        self._set_trainable(self.models.feature_module, False)
