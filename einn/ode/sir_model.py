import torch
import torch.nn as nn

from einn.model.interface.ode_solution import ODESolution
from einn.ode.base_ode_model import BaseODEModel


class SIRModel(BaseODEModel):
    """
    Implementation of the Susceptible-Infected-Recovered (SIR) epidemiological model.
    """
    def __init__(self):
        """
        Initializes the SIR model.
        """
        super().__init__()

    def init_params(self, param_dict: dict):
        """
        Initializes the raw parameters for beta and gamma using inverse tanh scaling.

        :param dict param_dict: Dictionary containing 'beta' and 'gamma' initial values.
        """
        beta_raw = self._inverse_tanh_init(target_val=param_dict.get("beta", 0.5))
        gamma_raw = self._inverse_tanh_init(target_val=param_dict.get("gamma", 0.5))

        self.raw_params = nn.Parameter(data=torch.tensor(data=[beta_raw, gamma_raw], dtype=torch.float32))

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Calculates the analytic derivatives of the SIR compartment states over time.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 3].
        :param bool detach_params: Whether to detach parameters from the computation graph.
        :return ODESolution: An object containing derivatives, scaled parameters, and a time placeholder.
        """
        params = self.get_scaled_params(detach=detach_params)

        beta = params[0]
        gamma = params[1]

        # Unpacks the [Batch, Seq_len, 3] states tensor along the last dimension (dim=-1).
        # This splits the 3 features into individual S, I, and R tensors of shape [Batch, Seq_len, 1].
        s, i_state, r = torch.split(tensor=states, split_size_or_sections=1, dim=-1)

        # System of differential equations
        ds_dt_val = -beta * s * i_state
        di_dt_val = beta * s * i_state - gamma * i_state
        dr_dt_val = gamma * i_state

        # Concatenating back into the [Batch, Seq_len, 3] format
        ds_dt_tensor = torch.cat(tensors=(ds_dt_val, di_dt_val, dr_dt_val), dim=-1)

        return ODESolution(
            ds_dt=ds_dt_tensor,
            params=params,
            t=torch.empty(size=(0,))  # Placeholder for interface compatibility since parameters are not time-dependent
        )
