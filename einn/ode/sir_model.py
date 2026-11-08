import torch
import torch.nn as nn

from einn.model.interface.ode_solution import ODESolution
from einn.ode.base_ode_model import BaseODEModel


class SIRModel(BaseODEModel):
    """
    Implementation of the Susceptible-Infected-Recovered (SIR) epidemiological model.
    """

    def __init__(self, population_n: float = 1.0, target_idx: int = 1):
        """
        Initializes the SIR model.

        :param float population_n: The total population (N). Defaults to 1.0 for normalized fractional populations.
        :param int target_idx: The index of the compartment that corresponds to the observed target data (e.g., 1 for Infected).
        """
        super().__init__(d_s=3, target_idx=target_idx)
        self.population_n = population_n

        self.monotonicity_indices = {
            "decreasing": [0],  # Susceptible
            "increasing": [2]  # Recovered
        }

    def init_params(self, param_dict: dict):
        """
        Initializes the raw parameters for beta and gamma using inverse tanh scaling.

        :param dict param_dict: Dictionary containing initial biological values for the parameters.
        """
        beta_raw = self._inverse_tanh_init(target_val=param_dict.get("beta", 0.5))
        gamma_raw = self._inverse_tanh_init(target_val=param_dict.get("gamma", 0.5))

        self.raw_params = nn.Parameter(data=torch.tensor(data=[beta_raw, gamma_raw], dtype=torch.float32))

    # TODO: ezt majd normálissá kell varázsolni
    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Pure mathematical implementation of the SIR differential equations.
        Allows explicit passing of states and parameters, enabling reusability across data generation,
        pre-calibration, and dynamic parameter predictions.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 3].
        :param torch.Tensor params: The current physical parameters [beta, gamma]. Shape: [Batch, Seq_len, 2] or [2].
        :return torch.Tensor: The computed time derivatives of the states. Shape: [Batch, Seq_len, 3].
        """

        # Using ellipsis [..., 0:1] ensures dimensional broadcasting works safely whether params
        # is a 1D tensor (static global parameter) or a 3D tensor (dynamic parameter series).
        beta = params[..., 0:1]
        gamma = params[..., 1:2]
        # Unpacks the states tensor along the last dimension
        s, i_state, r = torch.split(tensor=states, split_size_or_sections=1, dim=-1)

        # Standard epidemiological infection term normalized by the total population (N).
        infection_term = (beta * s * i_state) / self.population_n

        # System of differential equations
        ds_dt_val = -infection_term
        di_dt_val = infection_term - (gamma * i_state)
        dr_dt_val = gamma * i_state

        # Concatenating back into the appropriate format
        return torch.cat(tensors=(ds_dt_val, di_dt_val, dr_dt_val), dim=-1)

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Evaluates internal parameters and returns the structured ODESolution dataclass.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 3].
        :param bool detach_params: Whether to detach parameters from the computation graph.
        :return ODESolution: An object containing derivatives and the scaled parameters.
        """
        params = self.get_scaled_params(detach=detach_params)

        # A kód ismétlése helyett egyszerűen meghívjuk a forward metódust!
        ds_dt_tensor = self.forward(states=states, params=params)

        return ODESolution(
            ds_dt=ds_dt_tensor,
            params=params
        )