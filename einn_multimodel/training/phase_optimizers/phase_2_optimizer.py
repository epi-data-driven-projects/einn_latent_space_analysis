import itertools

import torch

from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer


class Phase2Optimizer(BasePhaseOptimizer):
    """Phase 2: train Time Modules, ODE parameters and shared Output Module."""

    def _init_optimizer(self) -> None:
        """Perform init optimizer as part of optimize the second stage of the original EINN objective.


        :return None: No return value; state or serialized artifacts are updated in place.
        """
        params = itertools.chain(
            self.models.time_modules.parameters(),
            self.models.ode_models.parameters(),
            self.models.output_module.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.learning_rate, amsgrad=True
        )

    def prepare_network_states(self) -> None:
        """Construct prepare network states using the configured model and experiment settings.


        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self._set_trainable(self.models.time_modules, True)
        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_models, True)
        self._set_trainable(self.models.feature_module, False)
