import torch
import torch.nn as nn

from einn.ode.base_ode_model import BaseODEModel
from einn.model.interface.ode_solution import ODESolution


class SISModel(BaseODEModel):
    """
    Implementation of the Susceptible-Infected-Susceptible (SIS) epidemiological model.
    Models diseases where infection does not confer long-lasting immunity.
    """

    def __init__(self, population_n: float = 1.0, target_idx: int = 1):
        """
        Initializes the SIS model.

        :param float population_n: Total population (N). Defaults to 1.0.
        :param int target_idx: Index for the observed target data (e.g., 1 for Infected).
        """
        super().__init__(d_s=2, target_idx=target_idx)
        self.population_n = population_n

        # The SIS model constantly recycles population, so neither compartment is strictly monotonic
        self.monotonicity_indices = {
            "decreasing": [],
            "increasing": []
        }

    def init_params(self, param_dict: dict):
        """
        Initializes the raw learnable parameters for beta and gamma using inverse tanh scaling.

        :param dict param_dict: Dictionary containing initial biological values.
                                Expected keys: "beta", "gamma".
        """
        beta_raw = self._inverse_tanh_init(target_val=param_dict.get("beta", 0.5))
        gamma_raw = self._inverse_tanh_init(target_val=param_dict.get("gamma", 0.5))

        self.raw_params = nn.Parameter(
            data=torch.tensor(data=[beta_raw, gamma_raw], dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Pure mathematical implementation of the SIS differential equations.

        :param torch.Tensor states: Compartment states. Shape: [Batch, Seq_len, 2].
        :param torch.Tensor params: Physical parameters [beta, gamma]. Shape: [Batch, Seq_len, 2] or [2].
        :return torch.Tensor: Computed time derivatives. Shape: [Batch, Seq_len, 2].
        """
        beta = params[..., 0:1]
        gamma = params[..., 1:2]

        s, i_state = torch.split(tensor=states, split_size_or_sections=1, dim=-1)

        infection_term = (beta * s * i_state) / self.population_n
        recovery_term = gamma * i_state

        ds_dt_val = -infection_term + recovery_term
        di_dt_val = infection_term - recovery_term

        return torch.cat(tensors=(ds_dt_val, di_dt_val), dim=-1)

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Evaluates internal parameters and returns the structured ODESolution dataclass.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 2].
        :param bool detach_params: If True, detaches the parameters from the PyTorch computation graph.
        :return ODESolution: A dataclass holding the computed derivatives (ds_dt) and the scaled parameters.
        """
        params = self.get_scaled_params(detach=detach_params)
        ds_dt_tensor = self.forward(states=states, params=params)

        return ODESolution(ds_dt=ds_dt_tensor, params=params)
