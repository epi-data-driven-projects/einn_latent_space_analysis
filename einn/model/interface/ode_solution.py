from dataclasses import dataclass

import torch


@dataclass
class ODESolution:
    """
    Data Transfer Object (DTO) containing the outputs of the physical ODE model.
    It encapsulates the analytical derivatives and the bounded physical parameters.

    - ds_dt (torch.Tensor): Analytical time derivatives of the compartments computed via differential equations.
                          Expected Shape: [Batch, Seq_len, d_s] (e.g., d_s=5 for SEIRM, d_s=3 for SIR).
    - params (torch.Tensor): Bounded, time-dependent physical parameters (e.g., beta, gamma).
                           Expected Shape: [Batch, Seq_len, d_p] (e.g., d_p=4 for SEIRM, d_p=2 for SIR).
    - t (torch.Tensor): The time vector corresponding to the solution (can be None if unused).
                      Expected Shape: [Batch, Seq_len, 1].
    """
    ds_dt: torch.Tensor
    params: torch.Tensor
    t: torch.Tensor
