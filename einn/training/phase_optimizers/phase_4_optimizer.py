import itertools

import torch

from einn.training.base_phase_optimizer import BasePhaseOptimizer


class Phase4Optimizer(BasePhaseOptimizer):
    """
    Optimizer for Phase 4: 'only out layer + ode params' (Gradient Trick).
    Responsible for the final fine-tuning step using the analytical ODE derivatives.
    Only the Output Layer and ODE Params are updated.
    """

    def _init_optimizer(self):
        """
        Initializes the PyTorch Adam optimizer for Phase 4.
        Chains the parameters of the ODE Model and the Output Module.
        """
        params = itertools.chain(
            self.models.ode_model.parameters(),
            self.models.output_module.parameters()
        )
        # The original EINN algorithm explicitly omits 'amsgrad=True' for this phase.
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

    def prepare_network_states(self):
        """
        Configures the gradient tracking and execution modes (train/eval) for all networks.
        Sets the Output module and ODE model to trainable.
        Strictly freezes the Time and Feature modules so latent embeddings act as constants.
        """
        self._set_trainable(self.models.time_module, trainable=False)
        self._set_trainable(self.models.feature_module, trainable=False)
        self._set_trainable(self.models.output_module, trainable=True)
        self._set_trainable(self.models.ode_model, trainable=True)
