"""
SIR ODE model embedded in a SEIR-compatible four-compartment [S, E, I, R] state space.

Allows SIR and SEIR to share a unified OutputModule with d_s = 4 and target_idx = 2
by representing the exposed compartment E as a zero-valued dummy state (thesis Section 2.4).
"""
from typing import Any

import torch
import torch.nn as nn

from einn.model.interface.ode_solution import ODESolution
from einn.ode.base_ode_model import BaseODEModel


class SIRSEIRCompatibleModel(BaseODEModel):
    """
    Mechanistic SIR dynamics formulated in the four-compartment [S, E, I, R] layout.

    Fixes compartment index 1 (E) to zero state and zero time derivative (dummy_state_indices = (1,))
    while evolving susceptible S (index 0), infected I (index 2), and recovered R (index 3)
    under standard SIR transmission and recovery rates (beta, gamma).
    """

    def __init__(self, population_n: float = 1.0, target_idx: int = 2) -> None:
        """
        Initialize the 4-state SIR model, dummy-state mask, and monotonicity constraints.

        :param float population_n: Total population N (1.0 when states represent population fractions).
        :param int target_idx: Index of the target epidemiological compartment (default 2 for I in [S, E, I, R]).
        :return None: Initializes base state dimension d_s = 4, dummy_state_indices, and monotonicity metadata.
        """
        super().__init__(d_s=4, target_idx=target_idx)
        self.population_n: float = float(population_n)

        # Exposed compartment E (index 1) is a structural dummy state for SIR-SEIR compatibility.
        self.dummy_state_indices: tuple[int, ...] = (1,)

        # Epidemiological monotonicity constraints: S is nonincreasing, R is nondecreasing.
        self.monotonicity_indices: dict[str, list[int]] = {
            "decreasing": [0],
            "increasing": [3],
        }

    def init_params(self, param_dict: dict[str, Any]) -> None:
        """
        Initialize unbounded raw learnable parameters from bounded biological (beta, gamma) guesses.

        :param dict[str, Any] param_dict: Initial parameter dictionary supporting 'beta'
            (default 0.5) and 'gamma' (default 0.1).
        :return None: Registers self.raw_params of shape (2,) as a trainable nn.Parameter.
        """
        beta_raw = self._inverse_tanh_init(
            target_val=float(param_dict.get("beta", 0.5))
        )
        gamma_raw = self._inverse_tanh_init(
            target_val=float(param_dict.get("gamma", 0.1))
        )
        self.raw_params = nn.Parameter(
            torch.tensor([beta_raw, gamma_raw], dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the [S, E, I, R] ODE right-hand side f_ODE(s, (beta, gamma)).

        Computes:
        dS/dt = -beta * S * I / N,
        dE/dt = 0,
        dI/dt = beta * S * I / N - gamma * I,
        dR/dt = gamma * I.

        :param torch.Tensor states: Compartment state tensor of shape (..., 4) ordered as [S, E, I, R].
        :param torch.Tensor params: Bounded parameter tensor of shape (..., 2) ordered as [beta, gamma].
        :return torch.Tensor: Time-derivative tensor ds/dt of shape (..., 4).
        """
        beta = params[..., 0:1]
        gamma = params[..., 1:2]

        susceptible = states[..., 0:1]
        exposed = states[..., 1:2]
        infected = states[..., 2:3]

        # Standard SIR incidence term with dummy zero derivative on E.
        infection = beta * susceptible * infected / self.population_n
        ds_dt = -infection
        de_dt = torch.zeros_like(exposed)
        di_dt = infection - gamma * infected
        dr_dt = gamma * infected

        return torch.cat((ds_dt, de_dt, di_dt, dr_dt), dim=-1)

    def get_derivatives(
        self, states: torch.Tensor, detach_params: bool
    ) -> ODESolution:
        """
        Retrieve bounded parameters and evaluate ODE time derivatives for the given states.

        :param torch.Tensor states: Compartment state tensor of shape (..., 4).
        :param bool detach_params: If True, detaches scaled parameters from the autograd graph.
        :return ODESolution: Dataclass containing ds_dt of shape (..., 4) and params of shape (..., 2).
        """
        params = self.get_scaled_params(detach=detach_params)
        return ODESolution(
            ds_dt=self.forward(states=states, params=params),
            params=params,
        )
