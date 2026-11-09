import torch
import torch.nn as nn

from einn.model.interface.ode_solution import ODESolution
from einn.ode.base_ode_model import BaseODEModel


class RostHungaryModel(BaseODEModel):
    """
    Röst Epidemic Model for Hungary (COVID-19).
    """

    def __init__(self, population_n: float = 1.0, target_idx: int = 15):
        """
        Initializes the comprehensive Röst Hungary ODE model.

        :param float population_n: Total population for normalization. Defaults to 1.0.
        :param int target_idx: Compartment index targeted by loss. Default is 15 (Cumulative Infected).
        """
        super().__init__(d_s=18, target_idx=target_idx)
        self.population_n = population_n

        self.monotonicity_indices = {
            "decreasing": [0],  # Susceptible
            "increasing": [13, 14, 15, 16, 17]  # Recovered, Dead, and all Cumulative trackers
        }

    def init_params(self, param_dict: dict):
        """
        Initializes all 14 raw learnable parameters using inverse tanh scaling.

        :param dict param_dict: Dictionary containing initial biological values for the parameters.
        """
        p_keys = [
            "beta", "inf_a", "susc", "alpha_l", "alpha_p", "p", "gamma_a",
            "gamma_s", "h", "xi", "gamma_h", "gamma_c", "mu", "gamma_cr"
        ]

        raw_vals = []
        for key in p_keys:
            val = param_dict.get(key, 0.1)
            raw_vals.append(self._inverse_tanh_init(target_val=val))

        self.raw_params = nn.Parameter(
            data=torch.tensor(data=raw_vals, dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Mathematical implementation of the 18 differential equations for the Röst model.

        :param torch.Tensor states: Compartment states. Shape: [Batch, Seq_len, 18].
        :param torch.Tensor params: Physical parameters. Shape: [Batch, Seq_len, 14] or [14].
        :return torch.Tensor: Computed time derivatives. Shape: [Batch, Seq_len, 18].
        """
        # Parameter extraction
        beta, inf_a, susc, alpha_l, alpha_p, p, gamma_a, gamma_s, h, xi, gamma_h, gamma_c, mu, gamma_cr = \
            torch.split(tensor=params, split_size_or_sections=1, dim=-1)

        # State extraction
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, d, inf, hosp, icu = \
            torch.split(tensor=states, split_size_or_sections=1, dim=-1)

        # Core Epidemiological Flows
        active_infectious = ip + inf_a * (ia1 + ia2 + ia3) + (is1 + is2 + is3)
        transmission = beta * active_infectious
        infection_flow = susc * (s / self.population_n) * transmission

        # Differential Equations
        ds = -infection_flow
        dl1 = infection_flow - 2 * alpha_l * l1
        dl2 = 2 * alpha_l * l1 - 2 * alpha_l * l2
        dip = 2 * alpha_l * l2 - alpha_p * ip
        dia1 = p * alpha_p * ip - 3 * gamma_a * ia1
        dia2 = 3 * gamma_a * ia1 - 3 * gamma_a * ia2
        dia3 = 3 * gamma_a * ia2 - 3 * gamma_a * ia3
        dis1 = (1 - p) * alpha_p * ip - 3 * gamma_s * is1
        dis2 = 3 * gamma_s * is1 - 3 * gamma_s * is2
        dis3 = 3 * gamma_s * is2 - 3 * gamma_s * is3

        dih = h * (1 - xi) * 3 * gamma_s * is3 - gamma_h * ih
        dic = h * xi * 3 * gamma_s * is3 - gamma_c * ic
        dicr = (1 - mu) * gamma_c * ic - gamma_cr * icr

        dr = 3 * gamma_a * ia3 + (1 - h) * 3 * gamma_s * is3 + gamma_h * ih + gamma_cr * icr
        dd = mu * gamma_c * ic

        # Cumulative Trackers
        dinf = 2 * alpha_l * l2
        dhosp = h * (1 - xi) * 3 * gamma_s * is3 + h * xi * 3 * gamma_s * is3
        dicu = h * xi * 3 * gamma_s * is3

        return torch.cat(tensors=(
            ds, dl1, dl2, dip, dia1, dia2, dia3, dis1, dis2, dis3,
            dih, dic, dicr, dr, dd, dinf, dhosp, dicu
        ), dim=-1)

    def get_derivatives(self, states: torch.Tensor, detach_params: bool) -> ODESolution:
        """
        Evaluates internal parameters and returns the structured ODESolution dataclass.

        :param torch.Tensor states: The current compartment states. Shape: [Batch, Seq_len, 18].
        :param bool detach_params: If True, detaches the parameters from the PyTorch computation graph.
        :return ODESolution: A dataclass holding the computed derivatives (ds_dt) and the scaled parameters.
        """
        params = self.get_scaled_params(detach=detach_params)
        ds_dt_tensor = self.forward(states=states, params=params)

        return ODESolution(ds_dt=ds_dt_tensor, params=params)
