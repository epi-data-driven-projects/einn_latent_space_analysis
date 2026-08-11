from abc import ABC, abstractmethod

import torch

from einn.training.einn_forward_engine import EINNForwardEngine
from einn.loss.einn_loss import EINNLoss
from einn.model.interface.phase_context import PhaseContext


class BasePhaseOptimizer(ABC):
    """
    Abstract base class for all training phase optimizers.
    Defines the contract for preparing network states (freezing/unfreezing)
    and executing the forward/backward passes.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, forward_engine: EINNForwardEngine, loss_calculator: EINNLoss) -> None:
        """
        Initializes the base phase optimizer.

        Args:
            optimizer (torch.optim.Optimizer): The PyTorch optimizer (e.g., Adam) assigned to this phase.
            forward_engine (EINNForwardEngine): Engine for handling autograd and forward passes.
            loss_calculator (EINNLoss): Module for computing the aggregated loss.
        """
        self.optimizer = optimizer
        self.forward_engine = forward_engine
        self.loss_calculator = loss_calculator

    @abstractmethod
    def prepare_network_states(self, context: PhaseContext) -> None:
        """
        Prepares the neural network modules for the specific training phase.
        Responsible for explicitly freezing and unfreezing weights and setting train/eval modes.

        Args:
            context (PhaseContext): The current training context holding model references.
        """
        pass

    @abstractmethod
    def step(self, context: PhaseContext) -> float:
        """
        Executes a single optimization step (forward pass, loss computation, backpropagation).

        Args:
            context (PhaseContext): The current training context holding data (X, y, t) and models.

        Returns:
            float: The scalar value of the calculated total loss for logging.
        """
        pass
