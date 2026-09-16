import itertools

import torch

from einn.training.base_phase_optimizer import BasePhaseOptimizer


class PhaseFourOptimizer(BasePhaseOptimizer):
    """
    Optimizer for Phase 4: 'only out layer + ode params' (Gradient Trick).
    Updates: Output Layer + ODE Params.
    """

    def _init_optimizer(self) -> None:
        params = itertools.chain(
            self.models.ode_model.parameters(),
            self.models.output_module.parameters()
        )
        # Original code explicitly omitted amsgrad=True for this phase
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

    def prepare_network_states(self):
        self._set_trainable(self.models.time_module, False)
        self._set_trainable(self.models.feature_module, False)

        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_model, True)
