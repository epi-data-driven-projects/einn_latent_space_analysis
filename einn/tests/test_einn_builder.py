import pytest
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.model.einn_builder import EINNBuilder
from einn.ode.sir_model import SIRModel


@pytest.fixture
def base_model_config() -> EINNModelConfig:
    """
    Fixture providing a base EINNModelConfig for testing architecture dimensions.
    """
    config = EINNModelConfig(
        d_x=12,
        d_e=24,
        d_s=3,
        d_p=2,
        time_mapping_size=30,
        feature_rnn_out=64,
        feature_n_layers=2
    )
    return config


@pytest.fixture
def base_train_config() -> EINNTrainConfig:
    """
    Fixture providing a base EINNTrainConfig mapped to CPU for testing.
    """
    config = EINNTrainConfig(
        device='cpu'
    )
    return config


@pytest.fixture
def base_calibration() -> dict:
    """
    Fixture providing dummy calibration parameters.
    """
    return {"beta": 0.4, "gamma": 0.2, "alpha": 0.1, "mu": 0.05}


def test_builder_instantiates_sir_model_and_networks(
        base_model_config: EINNModelConfig,
        base_train_config: EINNTrainConfig,
        base_calibration: dict
):
    """
    Tests if the EINNBuilder correctly instantiates the SIR model and all neural modules
    with the explicitly configured hyperparameters.

    :param EINNModelConfig base_model_config: Fixture providing the model configuration.
    :param EINNTrainConfig base_train_config: Fixture providing the training configuration.
    :param dict base_calibration: Fixture providing calibration data.
    """
    models = EINNBuilder.build_einn(
        model_config=base_model_config,
        train_config=base_train_config,
        param_calibration=base_calibration,
        model_type="SIR"
    )

    # Verifying ODE
    assert isinstance(models.ode_model, SIRModel), "Builder failed to instantiate SIRModel."

    # Verifying OutputModule
    assert models.output_module.net[0].in_features == 24, "OutputModule input dimension mismatch."
    assert models.output_module.net[-1].out_features == 3, "OutputModule output dimension mismatch."

    # FeatureModule Encoder dimensions
    assert models.feature_module.encoder.enc_rnn.hidden_size == 32, \
        "RNN hidden size mismatch (should be rnn_out // 2 for bidirectional)."
    assert models.feature_module.encoder.enc_rnn.num_layers == 2, "FeatureModule n_layers mismatch."


def test_builder_unsupported_model_raises_error(
        base_model_config: EINNModelConfig,
        base_train_config: EINNTrainConfig,
        base_calibration: dict
):
    """
    Tests if the Builder raises a ValueError for an unknown ODE model type.

    :param EINNModelConfig base_model_config: Fixture providing the model configuration.
    :param EINNTrainConfig base_train_config: Fixture providing the training configuration.
    :param dict base_calibration: Fixture providing calibration data.
    """
    with pytest.raises(expected_exception=ValueError, match="Unsupported ODE model type"):
        EINNBuilder.build_einn(
            model_config=base_model_config,
            train_config=base_train_config,
            param_calibration=base_calibration,
            model_type="UNKNOWN"
        )


def test_builder_device_allocation(
        base_model_config: EINNModelConfig,
        base_train_config: EINNTrainConfig,
        base_calibration: dict
):
    """
    Tests if the EINNBuilder correctly pushes all modules and parameters to the specified device.

    :param EINNModelConfig base_model_config: Fixture providing the model configuration.
    :param EINNTrainConfig base_train_config: Fixture providing the training configuration.
    :param dict base_calibration: Fixture providing calibration data.
    """
    base_train_config.device = 'cpu'

    models = EINNBuilder.build_einn(
        model_config=base_model_config,
        train_config=base_train_config,
        param_calibration=base_calibration,
        model_type="SIR"
    )

    assert next(models.time_module.parameters()).device.type == 'cpu', "TimeModule is not on the correct device."
    assert next(models.feature_module.parameters()).device.type == 'cpu', "FeatureModule is not on the correct device."
    assert next(models.output_module.parameters()).device.type == 'cpu', "OutputModule is not on the correct device."
    assert models.ode_model.raw_params.device.type == 'cpu', "ODE Model parameters are not on the correct device."


def test_builder_seed_reproducibility(
        base_model_config: EINNModelConfig,
        base_train_config: EINNTrainConfig,
        base_calibration: dict
):
    """
    Tests if passing the same seed yields identically initialized stochastic components (e.g., Fourier mapping),
    while different seeds yield different initializations.

    :param EINNModelConfig base_model_config: Fixture providing the model configuration.
    :param EINNTrainConfig base_train_config: Fixture providing the training configuration.
    :param dict base_calibration: Fixture providing calibration data.
    """
    models_seed_42_a = EINNBuilder.build_einn(
        model_config=base_model_config,
        train_config=base_train_config,
        param_calibration=base_calibration,
        model_type="SIR",
        seed=42
    )
    models_seed_42_b = EINNBuilder.build_einn(
        model_config=base_model_config,
        train_config=base_train_config,
        param_calibration=base_calibration,
        model_type="SIR",
        seed=42
    )
    models_seed_99 = EINNBuilder.build_einn(
        model_config=base_model_config,
        train_config=base_train_config,
        param_calibration=base_calibration,
        model_type="SIR",
        seed=99
    )

    b_matrix_42_a = models_seed_42_a.time_module.b_gauss
    b_matrix_42_b = models_seed_42_b.time_module.b_gauss
    b_matrix_99 = models_seed_99.time_module.b_gauss

    assert torch.equal(input=b_matrix_42_a, other=b_matrix_42_b), \
        "Builder failed reproducibility! Models with the same seed must have identical initializations."

    assert not torch.equal(input=b_matrix_42_a, other=b_matrix_99), \
        "Builder failed randomness! Models with different seeds must have different initializations."
