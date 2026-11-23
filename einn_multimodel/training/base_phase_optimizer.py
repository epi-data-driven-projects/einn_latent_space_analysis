from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.loss.einn_loss import EINNLoss
from einn_multimodel.model.interface.einn_models import EINNModels
from einn_multimodel.model.interface.phase_context import PhaseContext
from einn_multimodel.training.einn_forward_engine import EINNForwardEngine


class BasePhaseOptimizer(ABC):
    """Base optimizer preserving independent Adam state for each EINN phase."""

    def __init__(
        self,
        models: EINNModels,
        config: EINNTrainConfig,
        engine: EINNForwardEngine,
        loss_calculator: EINNLoss,
    ):
        """Initialize BasePhaseOptimizer to store the configuration and initialize internal state.


        :param EINNModels models: The models containing the time, feature, output, and ODE modules.
        :param EINNTrainConfig config: Configuration object or mapping controlling this operation.
        :param EINNForwardEngine engine: Inference engine containing the trained neural-network modules.
        :param EINNLoss loss_calculator: Object that combines the configured training loss components.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.models = models
        self.config = config
        self.engine = engine
        self.loss_calculator = loss_calculator
        self.optimizer: torch.optim.Optimizer | None = None
        self._init_optimizer()

    @abstractmethod
    def _init_optimizer(self) -> None:
        """Initialize phase-specific trainable parameter groups and their optimizer.


        :return None: No return value; state or serialized artifacts are updated in place.
        """
        raise NotImplementedError

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        """Enable gradient updates only for modules active in this training phase.


        :param nn.Module module: Neural module whose gradient-update state is configured.
        :param bool trainable: Whether model parameters should receive gradients in the current phase.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        for parameter in module.parameters():
            parameter.requires_grad = trainable
        module.train(trainable)

    @abstractmethod
    def prepare_network_states(self) -> None:
        """Construct prepare network states using the configured model and experiment settings.


        :return None: No return value; state or serialized artifacts are updated in place.
        """
        raise NotImplementedError

    def step(self, context: PhaseContext) -> tuple[float, dict[str, float]]:
        """Run one optimization step on the current mini-batch.


        :param PhaseContext context: Current optimization-phase input tensors, masks, and model modules.
        :return tuple[float, dict[str, float]]: Computed result in the configured output representation.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer was not initialized.")
        self.optimizer.zero_grad(set_to_none=True)
        self.prepare_network_states()
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
