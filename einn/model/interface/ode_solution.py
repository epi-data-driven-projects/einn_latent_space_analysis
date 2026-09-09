from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ODESolution:
    """
    Data class representing the output interface of ODE models.
    :param torch.Tensor ds_dt: Analytical time derivatives of the compartments.
        Shape: [Batch, Seq_len, d_s] (e.g., d_s = 5 for SEIRM, d_s = 3 for SIR)
    :param torch.Tensor params: Bounded, time-dependent physical parameters (e.g., beta, gamma)
        Shape: [Batch, Seq_len, d_p] (e.g., d_p = 4 for SEIRM, d_p = 2 for SIR)
    :param torch.Tensor t: The time vector corresponding to the solution.
        Shape:[Batch, Seq_len, 1].
    """
    ds_dt: torch.Tensor
    params: torch.Tensor
    t: Optional[torch.Tensor] = None
