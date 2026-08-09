import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from einn.model.interface.ode_solution import ODESolution


class BaseODEModel(nn.Module, ABC):
    """Minden ODE (fizikai) modell absztrakt ősosztálya."""

    def __init__(self):
        super(BaseODEModel, self).__init__()

    @abstractmethod
    def get_derivatives(self, states: torch.Tensor, params: torch.Tensor) -> ODESolution:
        """
        Kiszámolja az ODE deriváltakat az adott állapotok és paraméterek alapján.
        """
        pass
