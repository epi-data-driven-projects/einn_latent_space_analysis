from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from einn.config.einn_train_config import EINNTrainConfig
from einn.loss.einn_loss import EINNLoss
from einn.model.interface.einn_models import EINNModels
from einn.model.interface.phase_context import PhaseContext
from einn.training.einn_forward_engine import EINNForwardEngine


class BasePhaseOptimizer(ABC):
    """
    Abstract base class for Phase Optimizers.
    Each child class maintains its own PyTorch Adam optimizer to preserve momentum states across the training epochs.
    """

    def __init__(
            self,
            models: EINNModels,
            config: EINNTrainConfig,
            engine: EINNForwardEngine,
            loss_calculator: EINNLoss
    ):
        self.models = models
        self.config = config
        self.engine = engine
        self.loss_calculator = loss_calculator

        # A child class should initialize this variable
        self.optimizer: torch.optim.Optimizer | None = None
        self._init_optimizer()

    @abstractmethod
    def _init_optimizer(self):
        """
        Abstract method where child classes must define which parameters their
        specific Adam optimizer will track and update.
        """
        pass

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool):
        """
        Helper method to freeze or unfreeze a neural network module's parameters
        and toggle its training/evaluation mode accordingly.

        :param nn.Module module: The PyTorch neural network module to be modified.
        :param bool trainable: If True, enables gradient computation (requires_grad=True)  and sets the module to
         training mode. If False, disables gradients and sets the module to evaluation mode.
        """
        for param in module.parameters():
            param.requires_grad = trainable

        if trainable:
            module.train()
        else:
            module.eval()

    @abstractmethod
    def prepare_network_states(self):
        """
        Abstract method to freeze/unfreeze specific networks just before the forward pass.
        """
        pass

    def step(self, context: PhaseContext) -> float:
        """
        Executes a full optimization iteration: Forward -> Loss -> Backward -> Step.

        :param PhaseContext context: The current phase state and data batch.
        :return float: The calculated scalar loss value for logging.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer was not initialized in the child class.")

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Freeze/Unfreeze according to phase rules
        self.prepare_network_states()

        # Forward pass
        network_outputs = self.engine.forward(context=context)

        # Calculate loss
        loss = self.loss_calculator(phase_context=context, network_outputs=network_outputs)

        # Backward pass
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss.item()
