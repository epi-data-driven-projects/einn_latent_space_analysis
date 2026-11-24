"""
Training schedule, loss-weighting, and contrastive-learning configuration for multi-model EINN.
"""
from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class EINNTrainConfig:
    """
    Training hyperparameters, four-phase schedule, and loss weights for multi-model EINN.

    Extends the single-model EINN training configuration with Supervised Contrastive (SupCon)
    loss settings (weight lambda_SupCon and temperature tau > 0) and a loss-masking option
    for mismatched-model control experiments (thesis Sections 2.2-2.4, Section 5, and
    Appendix Table 2).

    :param float learning_rate: Base learning rate shared by the four phase optimizers; must be positive.
    :param int future_steps: Number of extrapolation timesteps |T_future| beyond the observation
        window where L_ODE,future^T and L_ODE,future^F are evaluated.
    :param Dict[str, int] reps: Inner iteration counts for each training phase (phase_1 to phase_4).
    :param float early_stopping_min_delta: Minimum loss decrease required to register an improvement.
    :param int early_stopping_patience: Number of consecutive non-improving steps before stopping a phase early.
    :param bool early_stopping_percentage: If True, interprets early_stopping_min_delta as a relative
        percentage rather than an absolute difference.
    :param str device: Target PyTorch device string (defaults to cuda:0 if CUDA is available, else cpu).
    :param Dict[str, float] loss_weights: Scalar weights lambda_* in L_total = L_EINN + lambda_SupCon * L_SupCon:

        * data_T, data_F: Target observation MSE for active Time and shared Feature branches.
        * aux: Auxiliary reference-compartment MSE (L_aux) on the active Time branch.
        * ode_T, ode_F: In-window mechanistic ODE residual losses (L_ODE^T, L_ODE^F).
        * ode_future_T, ode_future_F: Extrapolation-horizon ODE residual losses.
        * mono: Epidemiological monotonicity penalty (L_Mono).
        * param: Temporal parameter smoothness penalty (L_Param; 0.0 for constant ODE parameters).
        * kd_target, kd_emb: Time-to-Feature knowledge distillation on states and latent embeddings.
        * ic, ic_F: Initial-condition MSE penalties at t = 0 for Time and Feature branches (Section 2.3).
        * supcon: Supervised Contrastive loss weight lambda_SupCon applied in Phase 3 (Section 2.4).
    :param float supcon_temperature: Positive temperature parameter tau > 0 scaling cosine logits
        in SupConLoss (smaller values sharpen separation between positive and negative pairs; Section 2.4).
    :param bool mask_mismatched_losses: If True, excludes samples whose true generating model differs
        from the assumed model label from Time, ODE, and knowledge-distillation losses, while keeping
        Feature data, initial-condition, and SupCon losses active.
    """

    # Optimization step size and extrapolation horizon length.
    learning_rate: float = 0.001
    future_steps: int = 30

    # Inner iteration counts for each of the four EINN training phases (Section 2.2).
    reps: Dict[str, int] = field(default_factory=lambda: {
        "phase_1": 300,
        "phase_2": 300,
        "phase_3": 300,
        "phase_4": 300,
    })

    # Early stopping thresholds applied within phase optimization loops.
    early_stopping_min_delta: float = 0.0
    early_stopping_patience: int = 10
    early_stopping_percentage: bool = False

    # Default compute device selection.
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Relative weights (lambda_*) for the composite multi-model EINN objective (Sections 2.2-2.4).
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "data_T": 1.0,
        "data_F": 1.0,
        "aux": 0.1,
        "ode_T": 10.0,
        "ode_F": 10.0,
        "ode_future_T": 10.0,
        "ode_future_F": 10.0,
        "mono": 1.0,
        "param": 0.001,
        "kd_target": 1.0,
        "kd_emb": 5.0,
        "ic": 10.0,
        "ic_F": 10.0,
        "supcon": 0.0,
    })

    # Temperature tau > 0 for L2-normalized pairwise similarities in SupConLoss (Section 2.4).
    supcon_temperature: float = 0.1

    # When True, masks Time/ODE/KD losses on samples with mismatched assumed vs. true model labels.
    mask_mismatched_losses: bool = False

    def __post_init__(self) -> None:
        """
        Validate numerical bounds and flag types after dataclass initialization.

        :return None: Validates all fields in place.
        :raises ValueError: If learning_rate or supcon_temperature is not strictly positive.
        :raises TypeError: If mask_mismatched_losses is not a bool.
        """
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.supcon_temperature <= 0:
            raise ValueError("supcon_temperature must be positive.")
        if not isinstance(self.mask_mismatched_losses, bool):
            raise TypeError("mask_mismatched_losses must be a bool.")
