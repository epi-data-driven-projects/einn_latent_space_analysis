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
        """
        Initializes the base phase optimizer.

        :param EINNModels models: The container holding all initialized neural networks and the ODE model.
        :param EINNTrainConfig config: The training configuration containing hyperparameters (e.g., learning rates).
        :param EINNForwardEngine engine: The core engine responsible for executing the forward pass and
                                         computing derivatives.
        :param EINNLoss loss_calculator: The module responsible for calculating the composite physics and data losses.
        """
        self.models = models
        self.config = config
        self.engine = engine
        self.loss_calculator = loss_calculator

        # A child class MUST initialize this specific Adam optimizer in _init_optimizer()
        self.optimizer: torch.optim.Optimizer = None
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

    def apply_grad_clip(self):
        """
        Applies gradient clipping to all network parameters globally.

        PINNs heavily rely on computing derivatives of the network
        outputs with respect to time (ds/dt). Early in training, when the network's predictions severely
        violate the strict ODE physics, the resulting loss gradients can become large. Clipping the L2 norm of
        the gradients to a maximum of 1.0 ensures numerical stability and prevents the optimizer
        from taking catastrophic steps.
        """
        torch.nn.utils.clip_grad_norm_(self.models.time_module.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.models.feature_module.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.models.output_module.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.models.ode_model.parameters(), max_norm=1.0)

    def step(self, context: PhaseContext) -> tuple[float, dict[str, float]]:
        """
        Executes a full optimization iteration: Forward -> Loss -> Backward -> Step.

        :param PhaseContext context: The current phase state and data batch.
        :return tuple[float, dict[str, float]]: The calculated scalar loss value for logging and the
         dictionary for loss components.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer was not initialized in the child class.")

        # Zero gradients
        self.optimizer.zero_grad()

        # Freeze/Unfreeze according to phase rules
        self.prepare_network_states()

        # Forward pass
        network_outputs = self.engine.forward(context=context)

        # Calculate loss
        total_loss, loss_dict = self.loss_calculator(phase_context=context, network_outputs=network_outputs)

        # Backward pass
        total_loss.backward()

        # Safeguard against exploding gradients
        self.apply_grad_clip()

        # Update weights
        self.optimizer.step()

        return total_loss.item(), loss_dict
