import torch
import torch.nn as nn

from einn.model.interface.ode_solution import ODESolution
from einn.ode.base_ode_model import BaseODEModel


class SEIRModel(BaseODEModel):
    """
    Implementation of the Susceptible-Exposed-Infected-Recovered (SEIR) model.
    Models diseases with a significant incubation period where individuals are
    infected but not yet infectious.
    """

    def __init__(self, population_n: float = 1.0, target_idx: int = 2):
        """
        Initializes the SEIR model.

        :param float population_n: Total population (N). Defaults to 1.0.
        :param int target_idx: Index for the observed target data (e.g., 2 for Infected).
        """
        super().__init__(d_s=4, target_idx=target_idx)
        self.population_n = population_n

        self.monotonicity_indices = {
            "decreasing": [0],  # Susceptible
            "increasing": [3]   # Recovered
        }

    def init_params(self, param_dict: dict):
        """
        Initializes the raw learnable parameters for beta, alpha, and gamma using inverse tanh scaling.

        :param dict param_dict: Dictionary containing initial biological values.
                                Expected keys: "beta" (infection), "alpha" (incubation), "gamma" (recovery).
        """
        beta_raw = self._inverse_tanh_init(target_val=param_dict.get("beta", 0.3))
        alpha_raw = self._inverse_tanh_init(target_val=param_dict.get("alpha", 0.2))
        gamma_raw = self._inverse_tanh_init(target_val=param_dict.get("gamma", 0.1))

        self.raw_params = nn.Parameter(
            data=torch.tensor(data=[beta_raw, alpha_raw, gamma_raw], dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Pure mathematical implementation of the SEIR differential equations.

        :param torch.Tensor states: Compartment states. Shape: [Batch, Seq_len, 4].
        :param torch.Tensor params: Physical parameters [beta, alpha, gamma]. Shape: [Batch, Seq_len, 3] or [3].
        :return torch.Tensor: Computed time derivatives. Shape: [Batch, Seq_len, 4].
        """
        beta = params[..., 0:1]
        alpha = params[..., 1:2]
        gamma = params[..., 2:3]

        s, e, i_state, r = torch.split(tensor=states, split_size_or_sections=1, dim=-1)

        infection_term = (beta * s * i_state) / self.population_n

        ds_dt_val = -infection_term
        de_dt_val = infection_term - (alpha * e)
        di_dt_val = (alpha * e) - (gamma * i_state)
        dr_dt_val = gamma * i_state

        return torch.cat(tensors=(ds_dt_val, de_dt_val, di_dt_val, dr_dt_val), dim=-1)

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Evaluates internal parameters and returns the structured ODESolution dataclass.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 4].
        :param bool detach_params: If True, detaches the parameters from the PyTorch computation graph.
        :return ODESolution: A dataclass holding the computed derivatives (ds_dt) and the scaled parameters.
        """
        params = self.get_scaled_params(detach=detach_params)
        ds_dt_tensor = self.forward(states=states, params=params)

        return ODESolution(ds_dt=ds_dt_tensor, params=params)
