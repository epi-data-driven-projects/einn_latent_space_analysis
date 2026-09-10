from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NetworkOutputs:
    """
    Data class container for all neural network predictions, embeddings,
    and calculated derivatives across different modules and time horizons.

    :param s_t: predicted compartment states from the Time module
    :param s_t_F: predicted compartment states from the Feature module
    :param e_t: latent embeddings from the Time module
    :param e_t_F: latent embeddings from the Feature module
    :param ds_dt_T_nn: empirical derivatives from the Time module (current steps)
    :param ds_dt_T_ode: analytical ODE derivatives from the Time module (current steps)
    :param dS_dt_T_ode: analytical ODE derivatives from the Time module (for monotonicity)
    :param ds_dt_future_T_nn: empirical derivatives from the Time module (future steps)
    :param ds_dt_future_T_ode: analytical ODE derivatives from the Time module (future steps)
    :param ds_dt_F_nn: empirical derivatives from the Feature module via gradient trick
    :param ds_dt_F_ode: analytical ODE derivatives from the Feature module
    :param ds_dt_future_F_nn: empirical derivatives from the Feature module (future steps)
    :param ds_dt_future_F_ode: analytical ODE derivatives from the Feature module (future steps)
    :param params: time-dependent physical parameters (omega_t)
    """
    s_t: Optional[torch.Tensor] = None
    s_t_F: Optional[torch.Tensor] = None
    e_t: Optional[torch.Tensor] = None
    e_t_F: Optional[torch.Tensor] = None

    ds_dt_T_nn: Optional[torch.Tensor] = None
    ds_dt_T_ode: Optional[torch.Tensor] = None
    dS_dt_T_ode: Optional[torch.Tensor] = None

    ds_dt_future_T_nn: Optional[torch.Tensor] = None
    ds_dt_future_T_ode: Optional[torch.Tensor] = None

    ds_dt_F_nn: Optional[torch.Tensor] = None
    ds_dt_F_ode: Optional[torch.Tensor] = None

    ds_dt_future_F_nn: Optional[torch.Tensor] = None
    ds_dt_future_F_ode: Optional[torch.Tensor] = None

    params: Optional[torch.Tensor] = None
