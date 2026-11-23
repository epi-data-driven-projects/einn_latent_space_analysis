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
        self.models = models
        self.config = config
        self.engine = engine
        self.loss_calculator = loss_calculator
        self.optimizer: torch.optim.Optimizer | None = None
        self._init_optimizer()

    @abstractmethod
    def _init_optimizer(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable
        module.train(trainable)

    @abstractmethod
    def prepare_network_states(self) -> None:
        raise NotImplementedError

    def apply_grad_clip(self) -> None:
        modules = (
            self.models.time_modules,
            self.models.feature_module,
            self.models.output_module,
            self.models.ode_models,
        )
        for module in modules:
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)

    def step(self, context: PhaseContext) -> tuple[float, dict[str, float]]:
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
        #self.apply_grad_clip()
        self.optimizer.step()
        return float(total_loss.detach().cpu()), loss_dict
