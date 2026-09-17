import itertools

import torch

from einn.training.base_phase_optimizer import BasePhaseOptimizer


class Phase3Optimizer(BasePhaseOptimizer):
    """
    Optimizer for Phase 3: 'feat and time jointly'.
    Updates: Feature Module + Output Layer + Time Module. (ODE params are NOT updated).
    """

    def _init_optimizer(self) -> None:
        params = itertools.chain(
            self.models.feature_module.parameters(),
            self.models.output_module.parameters(),
            self.models.time_module.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate, amsgrad=True)

    def prepare_network_states(self):
        self._set_trainable(self.models.time_module, True)
        self._set_trainable(self.models.feature_module, True)
        self._set_trainable(self.models.output_module, True)

        # Although ODE loss is calculated, ODE params are not updated by this Adam optimizer.
        # Setting requires_grad=False saves memory and computation.
        self._set_trainable(self.models.ode_model, False)
