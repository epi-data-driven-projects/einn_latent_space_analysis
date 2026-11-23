from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NetworkOutputs:
    """Predictions and derivative tensors produced by EINNForwardEngine."""

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
