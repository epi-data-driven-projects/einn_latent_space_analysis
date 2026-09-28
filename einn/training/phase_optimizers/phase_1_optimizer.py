import itertools

import torch

from einn.training.base_phase_optimizer import BasePhaseOptimizer


class Phase1Optimizer(BasePhaseOptimizer):
    """
    Optimizer for Phase 1: 'time solo'.
    Responsible for pre-training the Time Module and the Output Layer.
    The Feature Module and ODE parameters are completely frozen during this phase.
    """

    def _init_optimizer(self) -> None:
        """
        Initializes the PyTorch Adam optimizer for Phase 1.
        Chains the parameters of the Time Module and Output Module.

        :return None: Modifies the internal optimizer state.
        """
        params = itertools.chain(
            self.models.time_module.parameters(),
            self.models.output_module.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate, amsgrad=True)

    def prepare_network_states(self):
        """
        Configures the gradient tracking and execution modes (train/eval) for all networks.
        Sets Time and Output modules to trainable, while freezing ODE and Feature modules.
        """
        self._set_trainable(self.models.time_module, trainable=True)
        self._set_trainable(self.models.output_module, trainable=True)
        self._set_trainable(self.models.ode_model, trainable=False)
        self._set_trainable(self.models.feature_module, trainable=False)
