from typing import Any

import torch
from torch import nn

from einn.model.interface.ode_solution import ODESolution
from einn.ode.base_ode_model import BaseODEModel


class SIRSISCompatibleModel(BaseODEModel):
    """
    Mechanistic SIS epidemiological ODE model with a zero-valued dummy third compartment.

    Evolves susceptible S (index 0) and infected I (index 1) with transmission rate beta
    and recovery/re-susceptibility rate gamma, while keeping index 2 fixed at zero derivative
    (dummy_state_indices = (2,)) to align with SIR's three-compartment state space.
    """

    def __init__(self, population_n: float = 1.0, target_idx: int = 1) -> None:
        """
        Initialize the SIS ODE model, dummy-state mask, and monotonicity metadata.

        :param float population_n: Total population N (1.0 when states represent population fractions).
        :param int target_idx: Index of the target epidemiological compartment (default 1 for I).
        """
        super().__init__(d_s=3, target_idx=target_idx)
        self.population_n: float = float(population_n)

        # Index 2 is a fixed zero padding channel for compatibility with d_s = 3 in SIR-SIS runs.
        self.dummy_state_indices: tuple[int, ...] = (2,)

        # Reinfection allows S to increase after declining, so global monotonicity is not enforced.
        self.monotonicity_indices: dict[str, list[int]] = {
            "decreasing": [0],
            "increasing": [1],
        }

    def init_params(self, param_dict: dict[str, Any]) -> None:
        """
        Initialize unbounded raw learnable parameters from bounded biological (beta, gamma) guesses.

        :param dict[str, Any] param_dict: Initial parameter dictionary supporting 'beta'
            (default 0.4) and 'gamma' (default 0.1).
        """
        beta_raw = self._inverse_tanh_init(float(param_dict.get("beta", 0.4)))
        gamma_raw = self._inverse_tanh_init(float(param_dict.get("gamma", 0.1)))
        self.raw_params = nn.Parameter(
            torch.tensor([beta_raw, gamma_raw], dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the SIS ODE right-hand side f_ODE(s, (beta, gamma)).

        Computes:
        dS/dt = -beta * S * I / N + gamma * I,
        dI/dt = beta * S * I / N - gamma * I,
        dR_dummy/dt = 0.

        :param torch.Tensor states: Compartment state tensor of shape (..., 3) ordered as [S, I, dummy].
        :param torch.Tensor params: Bounded parameter tensor of shape (..., 2) ordered as [beta, gamma].
        :return torch.Tensor: Time-derivative tensor ds/dt of shape (..., 3).
        """
        beta = params[..., 0:1]
        gamma = params[..., 1:2]
        susceptible = states[..., 0:1]
        infected = states[..., 1:2]

        # Infection moves individuals S -> I; recovery returns them immediately I -> S.
        infection = beta * susceptible * infected / self.population_n
        recovery = gamma * infected

        ds_dt = -infection + recovery
        di_dt = infection - recovery
        dr_dt = torch.zeros_like(infected)
        return torch.cat((ds_dt, di_dt, dr_dt), dim=-1)

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Retrieve bounded parameters and evaluate SIS ODE time derivatives for the given states.

        :param torch.Tensor states: Compartment state tensor of shape (..., 3).
        :param bool detach_params: If True, detaches scaled parameters from the autograd graph.
        :return ODESolution: Dataclass containing ds_dt of shape (..., 3) and params of shape (..., 2).
        """
        params = self.get_scaled_params(detach=detach_params)
        return ODESolution(
            ds_dt=self.forward(states=states, params=params),
            params=params,
        )
