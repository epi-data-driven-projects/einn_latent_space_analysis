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
        """
        beta_raw = self._inverse_tanh_init(target_val=param_dict.get("beta", 0.5))
        gamma_raw = self._inverse_tanh_init(target_val=param_dict.get("gamma", 0.5))

        self.raw_params = nn.Parameter(data=torch.tensor(data=[beta_raw, gamma_raw], dtype=torch.float32))

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Pure mathematical implementation of the SIR differential equations.
        Allows explicit passing of states and parameters (used by DataGenerator and Calibrator).
        """
        # A [..., 0:1] szeletelés biztosítja, hogy a dimenziók megmaradjanak (broadcasting)
        beta = params[..., 0:1]
        gamma = params[..., 1:2]

        # Unpacks the states tensor along the last dimension
        s, i_state, r = torch.split(tensor=states, split_size_or_sections=1, dim=-1)

        # System of differential equations
        ds_dt_val = -beta * s * i_state
        di_dt_val = beta * s * i_state - gamma * i_state
        dr_dt_val = gamma * i_state

        # Concatenating back into the appropriate format
        return torch.cat(tensors=(ds_dt_val, di_dt_val, dr_dt_val), dim=-1)

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Legacy interface for the EINN training loop. Evaluates internal parameters and
        returns the structured ODESolution dataclass.
        """
        params = self.get_scaled_params(detach=detach_params)

        # A kód ismétlése helyett egyszerűen meghívjuk a forward metódust!
        ds_dt_tensor = self.forward(states=states, params=params)

        return ODESolution(
            ds_dt=ds_dt_tensor,
            params=params
        )