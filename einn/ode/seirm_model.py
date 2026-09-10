import torch
import torch.nn as nn

from einn.ode.base_ode_model import BaseODEModel
from einn.model.interface.ode_solution import ODESolution


class SEIRMModel(BaseODEModel):
    """
    Implementation of the Susceptible-Exposed-Infected-Recovered-Mortality (SEIRM) model.
    """
    def __init__(self, population_n: float = 1.0):
        """
        Initializes the SEIRM model.

        :param float population_n: The total population (N). Defaults to 1.0 for normalized fractions.
        """
        super().__init__()
        self.population_n = population_n

    def init_params(self, param_dict: dict):
        """
        Initializes the raw parameters for alpha, beta, gamma, and mu using inverse tanh scaling.

        :param dict param_dict: Dictionary containing initial values for the 4 parameters.
        """
        beta_raw = self._inverse_tanh_init(target_val=param_dict.get("beta", 0.3))
        alpha_raw = self._inverse_tanh_init(target_val=param_dict.get("alpha", 0.2))
        gamma_raw = self._inverse_tanh_init(target_val=param_dict.get("gamma", 0.1))
        mu_raw = self._inverse_tanh_init(target_val=param_dict.get("mu", 0.05))

        self.raw_params = nn.Parameter(
            data=torch.tensor(data=[beta_raw, alpha_raw, gamma_raw, mu_raw], dtype=torch.float32)
        )

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Calculates the analytic derivatives of the SEIRM compartment states over time.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 5].
        :param bool detach_params: Whether to detach parameters from the computation graph.
        :return ODESolution: An object containing derivatives, scaled parameters, and a time placeholder.
        """
        params = self.get_scaled_params(detach=detach_params)

        # `params` is a 1D tensor containing 4 elements
        beta = params[0]
        alpha = params[1]
        gamma = params[2]
        mu = params[3]

        # Unpacks the [Batch, Seq_len, 5] states tensor along the last dimension
        s, e, i_state, r, m = torch.split(tensor=states, split_size_or_sections=1, dim=-1)

        # System of differential equations
        # to make the NN predict lower numbers, we can make its prediction to be N-Susceptible
        infection_term = (beta * s * i_state) / self.population_n

        ds_dt_val = -infection_term
        de_dt_val = infection_term - (alpha * e)
        di_dt_val = (alpha * e) - (gamma * i_state) - (mu * i_state)
        dr_dt_val = gamma * i_state
        dm_dt_val = mu * i_state

        # Concatenating back into the [Batch, Seq_len, 5] format
        ds_dt_tensor = torch.cat(tensors=(ds_dt_val, de_dt_val, di_dt_val, dr_dt_val, dm_dt_val), dim=-1)

        return ODESolution(
            ds_dt=ds_dt_tensor,
            params=params
        )
