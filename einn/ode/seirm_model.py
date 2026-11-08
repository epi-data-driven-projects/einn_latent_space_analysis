import torch
import torch.nn as nn

from einn.ode.base_ode_model import BaseODEModel
from einn.model.interface.ode_solution import ODESolution


class SEIRMModel(BaseODEModel):
    """
    Implementation of the Susceptible-Exposed-Infected-Recovered-Mortality (SEIRM) model.
    """

    def __init__(self, population_n: float = 1.0, target_idx: int = 4):
        """
        Initializes the SEIRM model.

        :param float population_n: The total population (N). Defaults to 1.0 for normalized fractional populations.
        :param int target_idx: The index of the compartment that corresponds to the observed target data (e.g., 4 for Mortality).
        """
        super().__init__(d_s=5, target_idx=target_idx)
        self.population_n = population_n

        self.monotonicity_indices = {
            "decreasing": [0],  # Susceptible
            "increasing": [3, 4]  # Recovered, Mortality
        }

    def init_params(self, param_dict: dict):
        """
        Initializes the raw learnable parameters for alpha, beta, gamma, and mu using inverse tanh scaling.

        :param dict param_dict: Dictionary containing initial biological values for the parameters.
                                Expected keys: "beta" (infection rate), "alpha" (incubation rate),
                                "gamma" (recovery rate), "mu" (mortality rate).
        """
        beta_raw = self._inverse_tanh_init(target_val=param_dict.get("beta", 0.3))
        alpha_raw = self._inverse_tanh_init(target_val=param_dict.get("alpha", 0.2))
        gamma_raw = self._inverse_tanh_init(target_val=param_dict.get("gamma", 0.1))
        mu_raw = self._inverse_tanh_init(target_val=param_dict.get("mu", 0.05))

        self.raw_params = nn.Parameter(
            data=torch.tensor(data=[beta_raw, alpha_raw, gamma_raw, mu_raw], dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Pure mathematical implementation of the SEIRM differential equations.
        Allows explicit passing of states and parameters, enabling reusability across data generation,
        pre-calibration, and dynamic parameter predictions.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 5].
        :param torch.Tensor params: The current physical parameters [beta, alpha, gamma, mu]. Shape: [Batch, Seq_len, 4] or [4].
        :return torch.Tensor: The computed time derivatives of the states. Shape: [Batch, Seq_len, 5].
        """
        # Using ellipsis [..., 0:1] ensures dimensional broadcasting works safely whether params
        # is a 1D tensor (static global parameter) or a 3D tensor (dynamic parameter series).
        beta = params[..., 0:1]
        alpha = params[..., 1:2]
        gamma = params[..., 2:3]
        mu = params[..., 3:4]

        # Unpacks the states tensor along the last dimension
        s, e, i_state, r, m = torch.split(tensor=states, split_size_or_sections=1, dim=-1)

        infection_term = (beta * s * i_state) / self.population_n

        # System of differential equations
        ds_dt_val = -infection_term
        de_dt_val = infection_term - (alpha * e)
        di_dt_val = (alpha * e) - (gamma * i_state) - (mu * i_state)
        dr_dt_val = gamma * i_state
        dm_dt_val = mu * i_state

        return torch.cat(tensors=(ds_dt_val, de_dt_val, di_dt_val, dr_dt_val, dm_dt_val), dim=-1)

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Evaluates internal parameters and returns the structured ODESolution dataclass.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 5].
        :param bool detach_params: Whether to detach parameters from the computation graph.
        :return ODESolution: An object containing derivatives and the scaled parameters.
        """
        params = self.get_scaled_params(detach=detach_params)

        ds_dt_tensor = self.forward(states=states, params=params)

        return ODESolution(
            ds_dt=ds_dt_tensor,
            params=params
        )
