import itertools

import torch

from einn.training.base_phase_optimizer import BasePhaseOptimizer


class Phase3Optimizer(BasePhaseOptimizer):
    """
    Optimizer for Phase 3: 'feat and time jointly'.
    Responsible for Knowledge Distillation. The Feature Module, Output Layer,
    and Time Module are updated. ODE parameters are NOT updated.
    """

    def _init_optimizer(self):
        """
        Initializes the PyTorch Adam optimizer for Phase 3.
        Chains the parameters of the Feature Module, Output Module, and Time Module.
        """
        params = itertools.chain(
            self.models.feature_module.parameters(),
            self.models.output_module.parameters(),
            self.models.time_module.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate, amsgrad=True)

    def prepare_network_states(self):
        """
        Configures the gradient tracking and execution modes (train/eval) for all networks.
        Sets Time, Feature, and Output modules to trainable.
        Freezes the ODE model to save memory and computation, as its parameters are not updated.
        """
        self._set_trainable(self.models.time_module, trainable=True)
        self._set_trainable(self.models.feature_module, trainable=True)
        self._set_trainable(self.models.output_module, trainable=True)

        # Although ODE loss is calculated, ODE params are not updated by this Adam optimizer.
        # Setting requires_grad=False saves memory and computation.
        self._set_trainable(self.models.ode_model, trainable=False)
