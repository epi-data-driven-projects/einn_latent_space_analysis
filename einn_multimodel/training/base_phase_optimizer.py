"""
Abstract base optimizer for each stage of the four-phase multi-model EINN training schedule.

Maintains a dedicated optimizer instance per phase, toggles parameter gradients and
train/eval modes across EINNModels submodules, runs the forward/loss/backward/step cycle,
and guards against non-finite loss values (thesis Sections 2.2 and 2.4).
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.loss.einn_loss import EINNLoss
from einn_multimodel.model.interface.einn_models import EINNModels
from einn_multimodel.model.interface.phase_context import PhaseContext
from einn_multimodel.training.einn_forward_engine import EINNForwardEngine


class BasePhaseOptimizer(ABC):
    """
    Abstract base class for phase-specific optimizers in multi-model EINN training.

    Each phase subclass defines its own trainable parameter group in _init_optimizer() and
    module freeze/unfreeze policy in prepare_network_states(), preserving separate Adam
    moment buffers per training phase.
    """

    def __init__(
        self,
        models: EINNModels,
        config: EINNTrainConfig,
        engine: EINNForwardEngine,
        loss_calculator: EINNLoss,
    ) -> None:
        """
        Store references to the model container, training config, forward engine, and loss module.

        Immediately invokes _init_optimizer() so self.optimizer is ready before step() is called.

        :param EINNModels models: Container holding all per-model Time/ODE modules and shared
            Feature/Output modules.
        :param EINNTrainConfig config: Training configuration supplying learning_rate and phase settings.
        :param EINNForwardEngine engine: Forward pass engine computing predictions and derivatives.
        :param EINNLoss loss_calculator: Phase-aware multi-model composite loss module.
        """
        self.models: EINNModels = models
        self.config: EINNTrainConfig = config
        self.engine: EINNForwardEngine = engine
        self.loss_calculator: EINNLoss = loss_calculator
        self.optimizer: torch.optim.Optimizer | None = None
        self._init_optimizer()

    @abstractmethod
    def _init_optimizer(self) -> None:
        """
        Instantiate self.optimizer over the trainable parameter groups for this training phase.
        """
        raise NotImplementedError

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        """
        Set requires_grad and train/eval mode uniformly across all parameters of a submodule.

        :param nn.Module module: Target PyTorch module or ModuleDict to configure.
        :param bool trainable: If True, enables parameter gradients and sets module.train(True);
            if False, disables parameter gradients and sets module.train(False).
        """
        for parameter in module.parameters():
            parameter.requires_grad = trainable
        module.train(trainable)

    @abstractmethod
    def prepare_network_states(self) -> None:
        """
        Configure gradient flags and train/eval modes for all EINN submodules in this phase.
        """
        raise NotImplementedError

    def step(self, context: PhaseContext) -> tuple[float, dict[str, float]]:
        """
        Execute one zero-grad, forward, loss, backward, and parameter-update step on a mini-batch.

        Raises FloatingPointError if total_loss is NaN or Inf so unstable runs fail fast
        before corrupting model weights.

        :param PhaseContext context: Batch tensors, active phase number, and model labels.
        :return tuple[float, dict[str, float]]: Pair (total_loss_float, loss_dict) containing
            the detached scalar total loss and per-term weighted loss values.
        :raises RuntimeError: If self.optimizer has not been initialized.
        :raises FloatingPointError: If total_loss is non-finite (NaN or +/-Inf).
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer was not initialized.")

        # Clear gradients with set_to_none=True to avoid unnecessary zero-fill overhead.
        self.optimizer.zero_grad(set_to_none=True)

        # Apply phase-specific module trainability and train/eval modes before forward pass.
        self.prepare_network_states()

        # Run forward pass, evaluate phase-weighted composite loss, and check finiteness.
        network_outputs = self.engine.forward(context=context)
        total_loss, loss_dict = self.loss_calculator(
            phase_context=context,
            network_outputs=network_outputs,
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite loss detected in phase {context.phase_num}: {total_loss.detach().item()}"
            )

        total_loss.backward()
        self.optimizer.step()
        return float(total_loss.detach().cpu()), loss_dict
