from typing import Dict, Any

import torch

from einn.ode.base_ode_model import BaseODEModel
from einn.model.interface.ode_solution import ODESolution


class SIRModel(BaseODEModel):
    """
    SIR epidemiology model (Susceptible, Infected, Recovered).
    Parameter indices mapping:
        0: beta (infection rate)
        1: gamma (recovery rate)
    """

    def __init__(self, seq_len: int, pop_total: float = 1.0, device: str = 'cpu') -> None:
        """
        Initializes the SIR physical model.

        Args:
            seq_len (int): Length of the sequence (number of time steps).
            pop_total (float): Total population size.
            device (str): Compute device ('cpu' or 'cuda').
        """
        # Set d_p = 2 (beta, gamma)
        super(SIRModel, self).__init__(seq_len=seq_len, d_p=2, pop_total=pop_total, device=device)

    def init_params(self, param_dict: Dict[str, Any]) -> None:
        """
        Initializes the time-dependent parameters based on a calibration dictionary.

        Args:
            param_dict (Dict[str, Any]): Dictionary containing calibration values.
        """
        with torch.no_grad():
            for t_idx in range(self.seq_len):
                step_key = f'ode_{t_idx}'
                p_vals = param_dict.get(step_key, param_dict.get('default', param_dict))

                # Apply inverse tanh initialization for each parameter
                self.raw_params[0, t_idx, 0] = self._inverse_tanh_init(p_vals.get('beta', 0.2))
                self.raw_params[0, t_idx, 1] = self._inverse_tanh_init(p_vals.get('gamma', 0.5))

    def get_derivatives(self, states: torch.Tensor, detach_params: bool = False) -> ODESolution:
        """
        Computes SIR ODE derivatives based on current states and parameters.

        Args:
            states (torch.Tensor): Compartment states (S, I, R). Shape: [Batch, Seq_len, 3].
            detach_params (bool): If True, physical parameters are detached from the gradient graph.

        Returns:
            ODESolution: Dataclass containing derivatives (ds_dt) and scaled parameters.
                         ds_dt Shape: [Batch, Seq_len, 3].
        """
        # Shape: [Batch, Seq_len, 2]
        params = self.get_scaled_params(detach=detach_params)
        params = params.expand(states.size(0), -1, -1)

        beta = params[..., 0:1]
        gamma = params[..., 1:2]

        # Extract states. Shapes: [Batch, Seq_len, 1] for each.
        S, I, R = states[..., 0:1], states[..., 1:2], states[..., 2:3]

        # ODE calculations
        dSI = beta * S * I / self.pop_total
        dIR = gamma * I

        dS = -1.0 * dSI
        dI = dSI - dIR
        dR = dIR

        # Concatenate derivatives along the feature dimension
        # Output shape: [Batch, Seq_len, 3]
        ds_dt = torch.cat([dS, dI, dR], dim=-1)

        return ODESolution(ds_dt=ds_dt, params=params, t=None)
