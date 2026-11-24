"""
Model architecture, ODE dimension, and multi-model conditioning configuration for EINN.
"""
from dataclasses import dataclass


@dataclass
class EINNModelConfig:
    """
    Architectural hyperparameters and ODE dimensions for the multi-model EINN.

    Extends the single-model EINN configuration with multi-model routing, learnable
    model-label embeddings, optional L2 normalization of Feature embeddings, and an
    ablation switch for model-label conditioning (thesis Sections 2.4 and 4.2, Figures 9-10).

    :param int d_x: Number of input observation features per timestep; input sequences have
        shape (B, T_obs, d_x).
    :param int d_e: Shared latent embedding dimension for all Time modules (e_t^{T_m}) and
        the shared Feature module (e_t^F), producing tensors of shape (B, T, d_e).
    :param int d_s: Unified ODE compartment state dimension output by the shared OutputModule,
        yielding state tensors of shape (B, T, d_s) (padded when models have different sizes,
        e.g. SIR-SEIR in Section 2.4).
    :param int d_p: Number of learnable mechanistic ODE parameters Omega in R^{d_p}.
    :param int time_mapping_size: Frequency count m for the fixed Gaussian Fourier feature
        vector b_gauss in R^m; maps scalar time (B, T, 1) to (B, T, 2 * m) (Section 2.2, Fig. 6).
    :param float time_scale: Standard deviation for sampling b_gauss ~ N(0, time_scale^2).
    :param int feature_rnn_out: Total hidden output dimension of the FeatureModule GRU
        (must be even when feature_bidirectional is True; e.g. 40 total = 20/direction, Fig. 10).
    :param int feature_n_layers: Number of stacked GRU layers in FeatureModule.
    :param bool feature_bidirectional: If True, uses a bidirectional GRU in FeatureModule.
    :param float feature_dropout: Dropout probability between stacked GRU layers.
    :param int num_models: Number of epidemiological model families jointly represented
        (e.g. 2 for SIR-SIS or SIR-SEIR pairs; Section 2.4).
    :param int model_label_dim: Embedding dimension for the learnable static model-label
        token e_m of shape (B, model_label_dim) fused into the Feature encoder (default 5, Fig. 10).
    :param bool normalize_feature_embeddings: If True, L2-normalizes Feature embeddings e_t^F
        before OutputModule so OutputModule, knowledge distillation, and SupCon share the exact
        same normalized representation; if False, only SupCon normalizes internally (Section 2.4).
    :param bool condition_feature_on_true_label: If True, conditions the Feature encoder on
        the sample's true model label a_i; if False (ablation control in Section 4.2/5), passes
        a single constant learned token to the encoder while Time/ODE routing and SupCon still
        use the true label.
    """

    # Core data, latent-space, and unified ODE dimensions.
    d_x: int = 10
    d_e: int = 20
    d_s: int = 5
    d_p: int = 4

    # Fourier feature mapping parameters for each model-specific TimeModule (Section 2.2, Fig. 6).
    time_mapping_size: int = 20
    time_scale: float = 1.0

    # Recurrent encoder/decoder hyperparameters for the shared FeatureModule (Section 2.4, Fig. 10).
    feature_rnn_out: int = 40
    feature_n_layers: int = 1
    feature_bidirectional: bool = True
    feature_dropout: float = 0.0

    # Multi-model family count and static model-label embedding dimension (Table 1, Fig. 10).
    num_models: int = 1
    model_label_dim: int = 5

    # Optional L2 normalization of e_t^F before OutputModule (otherwise only SupCon normalizes).
    normalize_feature_embeddings: bool = False

    # Ablation flag: if False, Feature encoder receives a constant token while Time/ODE and SupCon keep true labels.
    condition_feature_on_true_label: bool = True

    def __post_init__(self) -> None:
        """
        Validate dimensional constraints and boolean flag types after dataclass initialization.

        :return None: Validates all fields in place.
        :raises ValueError: If any dimension is non-positive or feature_rnn_out is odd for a bidirectional GRU.
        :raises TypeError: If normalize_feature_embeddings or condition_feature_on_true_label is not a bool.
        """
        if min(self.d_x, self.d_e, self.d_s, self.d_p) <= 0:
            raise ValueError("Network dimensions must be positive.")
        if self.feature_rnn_out <= 0:
            raise ValueError("feature_rnn_out must be positive.")
        if self.feature_bidirectional and self.feature_rnn_out % 2:
            raise ValueError("feature_rnn_out must be even for a bidirectional GRU.")
        if not isinstance(self.normalize_feature_embeddings, bool):
            raise TypeError("normalize_feature_embeddings must be bool.")
        if not isinstance(self.condition_feature_on_true_label, bool):
            raise TypeError("condition_feature_on_true_label must be bool.")
        if self.num_models < 1 or self.model_label_dim < 1:
            raise ValueError("num_models and model_label_dim must be positive.")
