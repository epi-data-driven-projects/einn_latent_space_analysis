import torch
from torch import nn

from einn.model.interface.ode_solution import ODESolution
from einn.ode.base_ode_model import BaseODEModel


class SISModel(BaseODEModel):
    """Susceptible-Infected-Susceptible model with a fixed dummy R state."""

    def __init__(self, population_n: float = 1.0, target_idx: int = 1):
        """Initialize SISModel to evaluate the susceptible-infected-susceptible mechanistic ODE system.


        :param float population_n: Total modeled population, or one for population-fraction states.
        :param int target_idx: Index of the epidemiological target compartment.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        super().__init__(d_s=3, target_idx=target_idx)
        self.population_n = float(population_n)
        self.dummy_state_indices = (2,)
        # In SIS, S need not be globally monotone: recovered individuals return to S.
        self.monotonicity_indices = {"decreasing": [], "increasing": []}

    def init_params(self, param_dict: dict) -> None:
        """Perform init params as part of evaluate the susceptible-infected-susceptible mechanistic ODE system.


        :param dict param_dict: Named initial guesses for the biological parameters.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        beta_raw = self._inverse_tanh_init(float(param_dict.get("beta", 0.4)))
        gamma_raw = self._inverse_tanh_init(float(param_dict.get("gamma", 0.1)))
        self.raw_params = nn.Parameter(
            torch.tensor([beta_raw, gamma_raw], dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Evaluate evaluate the susceptible-infected-susceptible mechanistic ODE system for batched inputs.


        :param torch.Tensor states: Current compartment populations in model state order.
        :param torch.Tensor params: Mechanistic model parameters in the documented parameter order.
        :return torch.Tensor: Tensor containing the computed model outputs or time derivatives.
        """
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
