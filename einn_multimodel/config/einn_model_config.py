from dataclasses import dataclass


@dataclass
class EINNModelConfig:
    """Configuration of EINN neural-network dimensions and architecture.

    :param d_x: The d x used by this operation.
    :param d_e: The d e used by this operation.
    :param d_s: The d s used by this operation.
    :param d_p: The d p used by this operation.
    :param time_mapping_size: The time mapping size used by this operation.
    :param time_scale: The time scale used by this operation.
    :param feature_rnn_out: The feature rnn out used by this operation.
    :param feature_n_layers: The feature n layers used by this operation.
    :param feature_bidirectional: The feature bidirectional used by this operation.
    :param feature_dropout: The feature dropout used by this operation.
    :param num_models: The num models used by this operation.
    :param model_label_dim: The model label dim used by this operation.
    :param normalize_feature_embeddings: The normalize feature embeddings used by this operation.
    :param condition_feature_on_true_label: The condition feature on true label used by this operation.
    """

    d_x: int = 10
    d_e: int = 20
    d_s: int = 5
    d_p: int = 4

    time_mapping_size: int = 20
    time_scale: float = 1.0

    feature_rnn_out: int = 40
    feature_n_layers: int = 1
    feature_bidirectional: bool = True
    feature_dropout: float = 0.0

    num_models: int = 1
    model_label_dim: int = 5

    # Apply L2 normalization to each Feature embedding before OutputModule.
    # The same representation is used for SupCon and embedding distillation.
    normalize_feature_embeddings: bool = False
    # False: encoder receives one constant learned model token for every sample.
    # Time/ODE routing and SupCon STILL use the true model label.
    condition_feature_on_true_label: bool = True

    def __post_init__(self) -> None:
        """Validate consistency of the initialized dataclass fields.


        :return None: No return value; state or serialized artifacts are updated in place.
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