"""SIS compartment model exposed through a common three-state S/I/R interface."""

import torch
from torch import nn

from einn.model.interface.ode_solution import ODESolution
from einn.ode.base_ode_model import BaseODEModel


class SISModel(BaseODEModel):
    """Susceptible-Infected-Susceptible model with a fixed dummy R state."""

    def __init__(self, population_n: float = 1.0, target_idx: int = 1):
        super().__init__(d_s=3, target_idx=target_idx)
        self.population_n = float(population_n)
        self.dummy_state_indices = (2,)
        # In SIS, S need not be globally monotone: recovered individuals return to S.
        self.monotonicity_indices = {"decreasing": [], "increasing": []}

    def init_params(self, param_dict: dict) -> None:
        beta_raw = self._inverse_tanh_init(float(param_dict.get("beta", 0.4)))
        gamma_raw = self._inverse_tanh_init(float(param_dict.get("gamma", 0.1)))
        self.raw_params = nn.Parameter(
            torch.tensor([beta_raw, gamma_raw], dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        beta = params[..., 0:1]
        gamma = params[..., 1:2]
        susceptible = states[..., 0:1]
        infected = states[..., 1:2]

        infection = beta * susceptible * infected / self.population_n
        recovery = gamma * infected

        ds_dt = -infection + recovery
        di_dt = infection - recovery
        dr_dt = torch.zeros_like(infected)
        return torch.cat((ds_dt, di_dt, dr_dt), dim=-1)

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        params = self.get_scaled_params(detach=detach_params)
        return ODESolution(
            ds_dt=self.forward(states=states, params=params),
            params=params,
        )