"""
Phase 2 optimizer for multi-model EINN training.

Jointly optimizes all model-specific TimeModules, learnable mechanistic ODE parameters
(ode_models), and the shared OutputModule via Adam (amsgrad=True) while keeping
feature_module frozen (thesis Sections 2.2 and 2.4).
"""
import itertools

import torch

from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer


class Phase2Optimizer(BasePhaseOptimizer):
    """
    Phase 2 optimization stage: fit mechanistic ODE residuals and parameters on the Time branch.

    Unfreezes ode_models alongside time_modules and output_module so in-window and
    extrapolation ODE residual losses (L_ODE^T, L_ODE,future^T) and parameter smoothness
    regularization (L_Param) refine both the latent time trajectories and physical
    epidemiological parameters while feature_module remains frozen (thesis Section 2.2).
    """

    def _init_optimizer(self):
        """
        Configure Adam with AMSGrad over TimeModule, ODE model, and shared OutputModule parameters.
        """
        params = itertools.chain(
            self.models.time_modules.parameters(),
            self.models.ode_models.parameters(),
            self.models.output_module.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.learning_rate, amsgrad=True
        )

    def prepare_network_states(self):
        """
        Enable gradients on time_modules, output_module, and ode_models; freeze feature_module.
        """
        self._set_trainable(self.models.time_modules, True)
        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_models, True)
        self._set_trainable(self.models.feature_module, False)
