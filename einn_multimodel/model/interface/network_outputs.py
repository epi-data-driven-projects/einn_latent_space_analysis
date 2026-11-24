"""
Dataclass container for forward-pass predictions, embeddings, and derivatives in multi-model EINN.
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NetworkOutputs:
    """
    Predictions, latent embeddings, and time derivatives produced by EINNForwardEngine.

    Stores all tensors required by EINNLoss across the four optimization phases (thesis
    Sections 2.2 and 2.4). Fields that are inactive in the current phase remain None.

    :param Optional[torch.Tensor] s_t: Normalized compartment states predicted by the active
        TimeModule -> OutputModule branch, of shape (B, T, d_s).
    :param Optional[torch.Tensor] s_t_F: Normalized compartment states predicted by the shared
        FeatureModule -> OutputModule branch, of shape (B, T, d_s).
    :param Optional[torch.Tensor] e_t: Latent embeddings e_t^T from the active TimeModule,
        of shape (B, T, d_e).
    :param Optional[torch.Tensor] e_t_F: Latent embeddings e_t^F from the shared FeatureModule,
        of shape (B, T, d_e).
    :param Optional[torch.Tensor] ds_dt_T_nn: Autograd time derivative d(s_hat^T)/dt over the
        observed horizon, of shape (B, T, d_s).
    :param Optional[torch.Tensor] ds_dt_T_ode: Mechanistic ODE RHS f_ODE(s_hat^T, Omega) over
        the observed horizon, of shape (B, T, d_s).
    :param Optional[torch.Tensor] ds_dt_future_T_nn: Autograd time derivative d(s_hat^T)/dt
        over the extrapolation horizon, of shape (B, T_future, d_s).
    :param Optional[torch.Tensor] ds_dt_future_T_ode: Mechanistic ODE RHS f_ODE(s_hat^T, Omega)
        over the extrapolation horizon, of shape (B, T_future, d_s).
    :param Optional[torch.Tensor] ds_dt_F_nn: Gradient-trick time derivative J_out(e_t^F) * d(e_t^T)/dt
        for the Feature branch over the observed horizon (Section 2.2), of shape (B, T, d_s).
    :param Optional[torch.Tensor] ds_dt_F_ode: Mechanistic ODE RHS f_ODE(s_hat^F, Omega) over
        the observed horizon, of shape (B, T, d_s).
    :param Optional[torch.Tensor] ds_dt_future_F_nn: Gradient-trick time derivative for the
        Feature branch over the extrapolation horizon, of shape (B, T_future, d_s).
    :param Optional[torch.Tensor] ds_dt_future_F_ode: Mechanistic ODE RHS f_ODE(s_hat^F, Omega)
        over the extrapolation horizon, of shape (B, T_future, d_s).
    :param dict[str, torch.Tensor] | None params: Dictionary mapping each active model name to
        its learnable ODE parameter tensor Omega of shape (d_p,) or (B_model, T, d_p).
    """

    # Predicted compartment states and latent embeddings for Time (T) and Feature (F) branches.
    s_t: Optional[torch.Tensor] = None
    s_t_F: Optional[torch.Tensor] = None
    e_t: Optional[torch.Tensor] = None
    e_t_F: Optional[torch.Tensor] = None

    # Time-branch autograd and ODE-RHS derivatives on observed and future horizons (Phase 2+).
    ds_dt_T_nn: Optional[torch.Tensor] = None
    ds_dt_T_ode: Optional[torch.Tensor] = None
    ds_dt_future_T_nn: Optional[torch.Tensor] = None
    ds_dt_future_T_ode: Optional[torch.Tensor] = None

    # Feature-branch gradient-trick and ODE-RHS derivatives on observed and future horizons (Phase 4).
    ds_dt_F_nn: Optional[torch.Tensor] = None
    ds_dt_F_ode: Optional[torch.Tensor] = None
    ds_dt_future_F_nn: Optional[torch.Tensor] = None
    ds_dt_future_F_ode: Optional[torch.Tensor] = None

    # Per-model mechanistic ODE parameter tensors keyed by uppercase model name.
    params: dict[str, torch.Tensor] | None = None
