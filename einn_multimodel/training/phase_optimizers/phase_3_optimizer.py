"""
Phase 3 optimizer for multi-model EINN training.

Jointly optimizes feature_module, output_module, and time_modules via Adam (amsgrad=True)
while keeping ode_models frozen, enabling Feature-branch data fitting, Time-to-Feature
knowledge distillation, and Supervised Contrastive (SupCon) representation learning
(thesis Sections 2.2 and 2.4).
"""
import itertools

import torch

from einn_multimodel.training.base_phase_optimizer import BasePhaseOptimizer


class Phase3Optimizer(BasePhaseOptimizer):
    """
    Phase 3 optimization stage: train Feature, Output, and Time modules with SupCon and distillation.

    Unfreezes feature_module alongside time_modules and output_module while keeping ode_models
    frozen so the shared Feature encoder-decoder learns from observed histories, distills
    Time-branch states/embeddings, and structures the latent space via SupConLoss without
    perturbing calibrated ODE parameters (thesis Sections 2.2 and 2.4).
    """

    def _init_optimizer(self):
        """
        Configure Adam with AMSGrad over FeatureModule, OutputModule, and TimeModule parameters.
        """
        params = itertools.chain(
            self.models.feature_module.parameters(),
            self.models.output_module.parameters(),
            self.models.time_modules.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.learning_rate, amsgrad=True
        )

    def prepare_network_states(self):
        """
        Enable gradients on time_modules, output_module, and feature_module; freeze ode_models.
        """
        self._set_trainable(self.models.time_modules, True)
        self._set_trainable(self.models.output_module, True)
        self._set_trainable(self.models.ode_models, False)
        self._set_trainable(self.models.feature_module, True)
