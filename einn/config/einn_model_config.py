from dataclasses import dataclass


@dataclass
class EINNModelConfig:
    """
    Configuration data class for the EINN neural network architectures and ODE dimensions.
    Contains:
    :param int d_x: dimensionality of input data features
    :param int d_e: latent embedding dimension
    :param int d_s: number of ODE compartment states
    :param int d_p: number of physical parameters
    :param int time_mapping_size: frequency mapping size for TimeModule
    :param float time_scale: variance scale for the TimeModule frequencies
    :param int feature_rnn_out: hidden size for the FeatureModule RNNs
    :param int feature_n_layers: number of RNN layers in FeatureModule
    :param bool feature_bidirectional: whether the Encoder RNN is bidirectional
    :param float feature_dropout: dropout probability for FeatureModule
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
