"""SIR ODE in the SEIR-compatible four-compartment state representation."""

import torch
import torch.nn as nn
from einn.model.interface.ode_solution import ODESolution

from einn.ode.base_ode_model import BaseODEModel


class SIRSEIRCompatibleModel(BaseODEModel):
    """
    SIR dynamics represented in the common [S, E, I, R] state space.

    The exposed compartment E is a fixed dummy state with E(0)=0 and dE/dt=0.
    This lets SIR and SEIR share one OutputModule with d_s=4.
    """

    def __init__(self, population_n: float = 1.0, target_idx: int = 2):
        """Initialize SIRSEIRCompatibleModel to store the configuration and initialize internal state.


        :param float population_n: Total modeled population, or one for population-fraction states.
        :param int target_idx: Index of the epidemiological target compartment.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        super().__init__(d_s=4, target_idx=target_idx)
        self.population_n = float(population_n)
        self.dummy_state_indices = (1,)
        self.monotonicity_indices = {
            "decreasing": [0],
            "increasing": [3],
        }

    def init_params(self, param_dict: dict) -> None:
        """Create bounded, trainable physical ODE parameters from calibration values.


        :param dict param_dict: Named initial guesses for the biological parameters.
        :return None: No return value; state or serialized artifacts are updated in place.
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
        """Evaluate the original neural or mechanistic forward transformation for batched inputs.


        :param torch.Tensor states: Current compartment populations in model state order.
        :param torch.Tensor params: Mechanistic model parameters in the documented parameter order.
        :return torch.Tensor: Tensor containing the computed model outputs or time derivatives.
        """
        beta = params[..., 0:1]
        gamma = params[..., 1:2]

        susceptible = states[..., 0:1]
        exposed = states[..., 1:2]
        infected = states[..., 2:3]

        infection = beta * susceptible * infected / self.population_n
        ds_dt = -infection
        de_dt = torch.zeros_like(exposed)
        di_dt = infection - gamma * infected
        dr_dt = gamma * infected

        return torch.cat((ds_dt, de_dt, di_dt, dr_dt), dim=-1)

    def get_derivatives(
        self, states: torch.Tensor, detach_params: bool
    ) -> ODESolution:
        """Retrieve get derivatives from the configured model or saved dataset.


        :param torch.Tensor states: Current compartment populations in model state order.
        :param bool detach_params: Whether ODE parameters should be detached from the gradient graph.
        :return ODESolution: Computed result in the configured output representation.
        """
        params = self.get_scaled_params(detach=detach_params)
        return ODESolution(
            ds_dt=self.forward(states=states, params=params),
            params=params,
        )
