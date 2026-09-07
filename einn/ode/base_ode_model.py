from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from einn.model.interface.ode_solution import ODESolution


class BaseODEModel(nn.Module, ABC):
    """
    Abstract base class for all epidemiological ODE models.
    """
    def __init__(self):
        """
        Initializes the BaseODEModel.
        """
        super().__init__()

        self.raw_params = None

    @abstractmethod
    def init_params(self, param_dict: dict):
        """
        Initializes the ODE parameters (as nn.Parameter) based on configuration.

        :param dict param_dict: Dictionary containing initial parameter values and bounds.
        """
        pass

    @abstractmethod
    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Calculates the analytic derivatives of the ODE compartment states over time.

        :param torch.Tensor states: The current compartment states.
        :param bool detach_params: Whether to detach parameters from the computation graph.
        :return ODESolution: An object containing derivatives, scaled parameters, and time.
        """
        pass

    def get_scaled_params(self, detach: bool = False) -> torch.Tensor:
        """
        Scales the raw parameters into physically meaningful bounds (0 to 1) using the tanh trick.

        :param bool detach: If True, detaches the tensor from the autograd graph.
        :return torch.Tensor: Scaled parameter tensor.
        """
        if self.raw_params is None:
            raise ValueError("raw_params must be initialized via init_params() first.")

        params = self.raw_params.detach() if detach else self.raw_params

        # Mapping from (-inf, inf) to (0, 1) using tanh
        return (torch.tanh(input=params) + 1.0) / 2.0

    @staticmethod
    def _inverse_tanh_init(target_val: float) -> float:
        """
        Calculates the inverse transformation for initialization.
        Assumes the forward scaling is: (tanh(x) + 1) / 2 = target_val.

        :param float target_val: The target physical value (must be strictly between 0 and 1).
        :return float: The unconstrained raw value to initialize the parameter with.
        """
        val_tensor = torch.tensor(data=target_val, dtype=torch.float32)

        # Clamping avoids torch.atanh domain errors (infinite values)
        clamped_val = torch.clamp(input=val_tensor, min=0.0001, max=0.9999)
        tanh_val = 2.0 * clamped_val - 1.0

        return torch.atanh(input=tanh_val)
