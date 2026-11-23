from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NetworkOutputs:
    """Predictions and derivative tensors produced by EINNForwardEngine.

    :param s_t: Time-module prediction of all epidemiological compartments.
    :param s_t_F: Feature-module prediction of all epidemiological compartments.
    :param e_t: Time-module latent vector for each requested time coordinate.
    :param e_t_F: Feature-module latent vector for each requested time coordinate.
    :param ds_dt_T_nn: Autograd-derived Time-module state derivative on the original horizon.
    :param ds_dt_T_ode: ODE-right-hand-side Time-module derivative on the original horizon.
    :param ds_dt_future_T_nn: Autograd-derived Time-module derivative beyond the original horizon.
    :param ds_dt_future_T_ode: ODE-derived Time-module derivative beyond the original horizon.
    :param ds_dt_F_nn: Autograd-derived Feature-module state derivative on the original horizon.
    :param ds_dt_F_ode: ODE-derived Feature-module derivative on the original horizon.
    :param ds_dt_future_F_nn: Autograd-derived Feature-module state derivative on future time steps.
    :param ds_dt_future_F_ode: ODE-derived Feature-module derivative on future time steps.
    :param params: Mechanistic ODE parameter tensor or dictionary, depending on experiment family.
    """

    s_t: Optional[torch.Tensor] = None
    s_t_F: Optional[torch.Tensor] = None
    e_t: Optional[torch.Tensor] = None
    e_t_F: Optional[torch.Tensor] = None

    ds_dt_T_nn: Optional[torch.Tensor] = None
    ds_dt_T_ode: Optional[torch.Tensor] = None
    ds_dt_future_T_nn: Optional[torch.Tensor] = None
    ds_dt_future_T_ode: Optional[torch.Tensor] = None

    ds_dt_F_nn: Optional[torch.Tensor] = None
    ds_dt_F_ode: Optional[torch.Tensor] = None
    ds_dt_future_F_nn: Optional[torch.Tensor] = None
    ds_dt_future_F_ode: Optional[torch.Tensor] = None

    params: dict[str, torch.Tensor] | None = None
