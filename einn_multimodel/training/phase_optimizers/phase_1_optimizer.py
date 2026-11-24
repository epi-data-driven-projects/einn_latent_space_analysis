"""
Phase 1 optimizer for multi-model EINN training.

Trains all model-specific TimeModules and the shared OutputModule via Adam (amsgrad=True)
while freezing mechanistic ODE parameters and the shared FeatureModule (thesis Section 2.2
and 2.4).
"""
import itertools

import torch

from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer


class Phase1Optimizer(BasePhaseOptimizer):
    """
    Phase 1 optimization stage: fit TimeModules and shared OutputModule to data, aux, IC, and mono losses.

    Keeps ode_models and feature_module frozen so Fourier-encoded time embeddings e_t^{T_m} and
    the shared state decoder OutputModule first learn clean compartment trajectories before
    mechanistic ODE residual fitting begins in Phase 2 (thesis Section 2.2).
    """

    def _init_optimizer(self):
        """
        Configure Adam with AMSGrad over all TimeModule parameters and shared OutputModule parameters.
        """
        params = itertools.chain(
            self.models.time_modules.parameters(),
            self.models.output_module.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.learning_rate, amsgrad=True
        )

    def prepare_network_states(self):
        """
        Enable gradients on time_modules and output_module; freeze ode_models and feature_module.
        """
        self._set_trainable(self.models.time_modules, True)
        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_models, False)
        self._set_trainable(self.models.feature_module, False)
