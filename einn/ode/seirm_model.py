from typing import Dict, Any

import torch

from einn.ode.base_ode_model import BaseODEModel
from einn.model.interface.ode_solution import ODESolution


class SEIRMModel(BaseODEModel):
    """
    SEIRM epidemiology model (Susceptible, Exposed, Infected, Recovered, Mortality).
    Parameter indices mapping:
        0: alpha
        1: beta
        2: gamma
        3: mu
    """

    def __init__(self, seq_len: int, pop_total: float = 1.0, device: str = 'cpu') -> None:
        """
        Initializes the SEIRM physical model.

        Args:
            seq_len (int): Length of the sequence (number of time steps).
            pop_total (float): Total population size.
            device (str): Compute device ('cpu' or 'cuda').
        """
        # Set d_p = 4 (alpha, beta, gamma, mu)
        super(SEIRMModel, self).__init__(seq_len=seq_len, d_p=4, pop_total=pop_total, device=device)

    def init_params(self, param_dict: Dict[str, Any]) -> None:
        """
        Initializes the time-dependent parameters based on a calibration dictionary.

        Format expected:
            param_dict = {'ode_0': {'alpha': 0.2, ...}, 'ode_1': {...}}
            OR a default dict: {'alpha': 0.2, 'beta': 0.2, ...}

        Args:
            param_dict (Dict[str, Any]): Dictionary containing calibration values.
        """
        with torch.no_grad():
            for t_idx in range(self.seq_len):
                # Try to get time-specific params, otherwise fall back to 'default' or direct keys
                step_key = f'ode_{t_idx}'
                p_vals = param_dict.get(step_key, param_dict.get('default', param_dict))

                # Apply inverse tanh initialization for each parameter
                self.raw_params[0, t_idx, 0] = self._inverse_tanh_init(p_vals.get('alpha', 0.2))
                self.raw_params[0, t_idx, 1] = self._inverse_tanh_init(p_vals.get('beta', 0.2))
                self.raw_params[0, t_idx, 2] = self._inverse_tanh_init(p_vals.get('gamma', 0.5))
                self.raw_params[0, t_idx, 3] = self._inverse_tanh_init(p_vals.get('mu', 0.01))

    def get_derivatives(self, states: torch.Tensor, detach_params: bool = False) -> ODESolution:
        """
        Computes SEIRM ODE derivatives based on current states and parameters.

        Args:
            states (torch.Tensor): Compartment states (S, E, I, R, M). Shape: [Batch, Seq_len, 5].
            detach_params (bool): If True, physical parameters are detached from the gradient graph.

        Returns:
            ODESolution: Dataclass containing derivatives (ds_dt) and scaled parameters.
                         ds_dt Shape: [Batch, Seq_len, 5].
        """
        # Get scaled parameters (and optionally detach them from computational graph)
        # Shape: [1, Seq_len, 4]
        params = self.get_scaled_params(detach=detach_params)

        # Broadcast parameters to match Batch size if Batch > 1
        # Shape becomes: [Batch, Seq_len, 4]
        params = params.expand(states.size(0), -1, -1)

        alpha = params[..., 0:1]
        beta = params[..., 1:2]
        gamma = params[..., 2:3]
        mu = params[..., 3:4]

        # Extract states. Shapes: [Batch, Seq_len, 1] for each.
        S, E, I, R, M = states[..., 0:1], states[..., 1:2], states[..., 2:3], states[..., 3:4], states[..., 4:5]

        # ODE calculations based on original formulation
        dSE = beta * S * I / self.pop_total
        dEI = alpha * E
        dIR = gamma * I
        dIM = mu * I

        dS = -1.0 * dSE
        dE = dSE - dEI
        dI = dEI - dIR - dIM
        dR = dIR
        dM = dIM

        # Concatenate derivatives along the feature dimension
        # Output shape: [Batch, Seq_len, 5]
        ds_dt = torch.cat([dS, dE, dI, dR, dM], dim=-1)

        return ODESolution(ds_dt=ds_dt, params=params, t=None)
