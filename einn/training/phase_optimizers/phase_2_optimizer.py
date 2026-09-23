import itertools

import torch

from einn.training.base_phase_optimizer import BasePhaseOptimizer


class Phase2Optimizer(BasePhaseOptimizer):
    """
    Optimizer for Phase 2: 'time + ode'.
    Responsible for incorporating physical ODE parameters into the Time Module's training.
    The Feature Module remains frozen.
    """

    def _init_optimizer(self):
        """
        Initializes the PyTorch Adam optimizer for Phase 2.
        Chains the parameters of the Time Module, ODE Model, and Output Module.
        """
        params = itertools.chain(
            self.models.time_module.parameters(),
            self.models.ode_model.parameters(),
            self.models.output_module.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate, amsgrad=True)

    def prepare_network_states(self):
        """
        Configures the gradient tracking and execution modes (train/eval) for all networks.
        Sets Time, Output, and ODE modules to trainable, while freezing the Feature module.
        """
        self._set_trainable(self.models.time_module, trainable=True)
        self._set_trainable(self.models.output_module, trainable=True)
        self._set_trainable(self.models.ode_model, trainable=True)
        self._set_trainable(self.models.feature_module, trainable=False)
