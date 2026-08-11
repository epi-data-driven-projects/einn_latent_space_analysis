from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from einn.model.interface.ode_solution import ODESolution


class BaseODEModel(nn.Module, ABC):
    """
    Abstract base class for all physical ODE models (e.g., SEIRM, SIR).
    Handles the scaling, inverse-scaling, and device management of
    time-dependent parameters.
    """

    def __init__(self, seq_len: int, d_p: int, pop_total: float = 1.0, device: str = 'cpu') -> None:
        """
        Initializes the base ODE model.

        Args:
            seq_len (int): Length of the sequence (number of time steps).
            d_p (int): Number of time-dependent parameters.
            pop_total (float): Total population size for normalization.
            device (str): Compute device ('cpu' or 'cuda').
        """
        super(BaseODEModel, self).__init__()
        self.pop_total = pop_total
        self.seq_len = seq_len
        self.d_p = d_p
        self.device = torch.device(device)

        # Vectorized time-dependent parameters.
        # Shape: [1, Seq_len, d_p]
        # This replaces the need for hundreds of separate ModuleDict entries.
        # Initialized with zeros (which maps to 0.5 after scaled tanh) directly on the target device.
        self.raw_params = nn.Parameter(
            torch.zeros(1, seq_len, d_p, dtype=torch.float32, device=self.device),
            requires_grad=True
        )

    def get_scaled_params(self, detach: bool = False) -> torch.Tensor:
        """
        Applies the (tanh(x) + 1) / 2 transformation to bound parameters between [0, 1].

        Args:
            detach (bool): If True, detaches the parameters from the computational graph
                           (equivalent to ODE_detach in the original implementation).

        Returns:
            torch.Tensor: Bounded parameters of shape [1, Seq_len, d_p].
        """
        params = self.raw_params.detach() if detach else self.raw_params

        # Original EINN transformation: y = (tanh(x) + 1) * 0.5
        scaled_params = (torch.tanh(params) + 1.0) * 0.5
        return scaled_params

    def _inverse_tanh_init(self, target_val: float) -> float:
        """
        Calculates the inverse of the bounding function: x = arctanh(2y - 1).
        Used to initialize raw_params accurately from a target biological value.

        Args:
            target_val (float): The desired biological parameter value in [0, 1].

        Returns:
            float: The inverse mapped value to be stored in raw_params.
        """
        # Add epsilon to prevent infinity values at boundaries (0 or 1)
        eps = -1e-12 if target_val > 0.0 else 1e-12

        # Math: 1 / 0.5 * (val - 0) - 1  ---> 2 * val - 1
        inner_val = 2.0 * target_val - 1.0 + eps

        # Clip to avoid numerical instability in arctanh
        inner_val = np.clip(inner_val, -0.999999, 0.999999)
        return float(np.arctanh(inner_val))

    @abstractmethod
    def init_params(self, param_dict: Dict[str, Any]) -> None:
        """
        Abstract method to initialize parameters from an external dictionary (e.g., JSON).
        Must be implemented by child classes to map dictionary keys to tensor indices.
        """
        pass

    @abstractmethod
    def get_derivatives(self, states: torch.Tensor, detach_params: bool = False) -> ODESolution:
        """
        Abstract method to calculate differential equations.
        Supports dynamic batch sizes via broadcasting.

        Args:
            states (torch.Tensor): Predicted compartment states. Shape: [Batch, Seq_len, d_s].
            detach_params (bool): If True, disconnects parameters from the graph.

        Returns:
            ODESolution: Dataclass containing derivatives and scaled parameters.
        """
        pass
