"""
Phase 4 optimizer for multi-model EINN training.

Fine-tunes mechanistic ODE parameters (ode_models) and the shared OutputModule via Adam
while freezing time_modules and feature_module during Feature-branch ODE residual
optimization (thesis Sections 2.2 and 2.4).
"""
import itertools

import torch

from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer


class Phase4Optimizer(BasePhaseOptimizer):
    """
    Phase 4 optimization stage: fine-tune ODE models and shared OutputModule on Feature ODE loss.

    Freezes time_modules and feature_module so their learned temporal and latent representations
    remain fixed while ode_models and output_module adapt to in-window and extrapolation
    Feature-branch ODE residual losses (L_ODE^F, L_ODE,future^F) evaluated via the gradient
    trick (thesis Section 2.2). Note that Phase 4 uses standard Adam (without amsgrad=True).
    """

    def _init_optimizer(self) -> None:
        """
        Configure standard Adam over ODE model and shared OutputModule parameters.

        :return None: Initializes self.optimizer with learning rate self.config.learning_rate.
        """
        params = itertools.chain(
            self.models.ode_models.parameters(),
            self.models.output_module.parameters(),
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

    def prepare_network_states(self) -> None:
        """
        Enable gradients on ode_models and output_module; freeze time_modules and feature_module.

        :return None: Updates requires_grad flags across EINNModels submodules in place.
        """
        self._set_trainable(self.models.time_modules, False)
        self._set_trainable(self.models.feature_module, False)
        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_models, True)
