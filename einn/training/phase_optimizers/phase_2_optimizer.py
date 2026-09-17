import itertools

import torch

from einn.training.base_phase_optimizer import BasePhaseOptimizer


class Phase2Optimizer(BasePhaseOptimizer):
    """
    Optimizer for Phase 2: 'time + ode'.
    Updates: Time Module + Output Layer + ODE Params.
    """

    def _init_optimizer(self):
        params = itertools.chain(
            self.models.time_module.parameters(),
            self.models.ode_model.parameters(),
            self.models.output_module.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate, amsgrad=True)

    def prepare_network_states(self):
        self._set_trainable(self.models.time_module, True)
        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_model, True)

        self._set_trainable(self.models.feature_module, False)
