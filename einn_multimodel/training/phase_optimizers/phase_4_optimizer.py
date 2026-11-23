import itertools

import torch

from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer


class Phase4Optimizer(BasePhaseOptimizer):
    """Phase 4: freeze Time/Feature modules and fine-tune ODE + Output modules."""

    def _init_optimizer(self) -> None:
        params = itertools.chain(
            self.models.ode_models.parameters(),
            self.models.output_module.parameters(),
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

    def prepare_network_states(self) -> None:
        self._set_trainable(self.models.time_modules, False)
        self._set_trainable(self.models.feature_module, False)
        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_models, True)
