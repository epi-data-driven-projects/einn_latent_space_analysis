import numpy as np
import pytest

from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel


@pytest.fixture
def sir_generator() -> SyntheticDataGenerator:
    """
    Fixture providing a data generator initialized with an SIR model.
    Sequence length is set to 200 days.

    :return SyntheticDataGenerator: The initialized generator instance.
    """
    ode_model = SIRModel(population_n=1.0, target_idx=1)
    return SyntheticDataGenerator(true_ode_model=ode_model, seq_len=200)


@pytest.fixture
def seirm_generator() -> SyntheticDataGenerator:
    """
    Fixture providing a data generator initialized with a SEIRM model.
    Sequence length is set to 150 days.

    :return SyntheticDataGenerator: The initialized generator instance.
    """
    ode_model = SEIRMModel(population_n=1.0, target_idx=4)
    return SyntheticDataGenerator(true_ode_model=ode_model, seq_len=150)


def test_generator_output_shapes_and_types(sir_generator: SyntheticDataGenerator):
    """
    Verifies that the generated synthetic data dictionary contains NumPy arrays
    with the mathematically correct dimensions across the sequence length and feature sizes.

    :param SyntheticDataGenerator sir_generator: The SIR generator fixture.
    """
    # Standard SIR parameters: Initial [S, I, R], True Params [beta, gamma]
    initial_states = [0.99, 0.01, 0.0]
    true_params = [0.3, 0.1]

    data_dict = sir_generator.generate(initial_states=initial_states, true_params=true_params)

    assert "x_raw" in data_dict
    assert "y_raw" in data_dict
    assert "true_latent_states" in data_dict

    x_raw = data_dict["x_raw"]
    y_raw = data_dict["y_raw"]
    states = data_dict["true_latent_states"]

    # Validations:
    # seq_len = 200
    # d_x = 4 exogenous features + 1 autoregressive target feature = 5
    # d_s = 3 (S, I, R)
    assert isinstance(x_raw, np.ndarray), "x_raw must be a NumPy array."
    assert x_raw.shape == (200, 5), f"Expected x_raw shape (200, 5), got {x_raw.shape}"

    assert isinstance(y_raw, np.ndarray), "y_raw must be a NumPy array."
    assert y_raw.shape == (200, 1), f"Expected y_raw shape (200, 1), got {y_raw.shape}"

    assert isinstance(states, np.ndarray), "states must be a NumPy array."
    assert states.shape == (200, 3), f"Expected states shape (200, 3), got {states.shape}"


def test_generator_biological_constraints(seirm_generator: SyntheticDataGenerator):
    """
    Ensures that the injected observational noise does not violate basic biological realities
    (e.g., compartment populations cannot drop below absolute zero).

    :param SyntheticDataGenerator seirm_generator: The SEIRM generator fixture.
    """
    initial_states = [0.99, 0.0, 0.01, 0.0, 0.0]
    true_params = [0.3, 0.2, 0.1, 0.05]

    # Injecting massive noise to force potential negative values
    data_dict = seirm_generator.generate(
        initial_states=initial_states,
        true_params=true_params,
        noise_std=5.0
    )

    y_raw = data_dict["y_raw"]

    assert np.min(y_raw) >= 0.0, \
        "Biological constraint violated: observed targets dropped below zero despite clipping."


def test_feature_correlation_statistical_accuracy(sir_generator: SyntheticDataGenerator):
    """
    Advanced statistical test: Extracts the generated exogenous features and mathematically
    calculates their actual Pearson correlation coefficient against the pure target variable.
    Proves that the internal _create_correlated_feature math works correctly.

    :param SyntheticDataGenerator sir_generator: The SIR generator fixture.
    """
    initial_states = [0.99, 0.01, 0.0]
    true_params = [0.4, 0.1]

    # Extreme correlation targets to easily verify the signal processing
    target_correlations = {
        'mobility': -0.9,  # Almost perfectly inverse
        'temperature': 0.0,  # Completely uncorrelated
        'testing': 0.8,  # Highly correlated
        'stringency': 0.99  # Almost perfectly correlated
    }

    # Generate data with low extra noise so correlations remain clear and calculable
    data_dict = sir_generator.generate(
        initial_states=initial_states,
        true_params=true_params,
        noise_std=0.01,
        correlations=target_correlations
    )

    x_raw = data_dict["x_raw"]
    pure_target = data_dict["true_latent_states"][:, sir_generator.true_ode_model.target_idx]

    # Extract specific features based on concatenation order:
    # [Mobility(0), Temperature(1), Testing(2), Stringency(3), Target(4)]
    mobility = x_raw[:, 0]
    stringency = x_raw[:, 3]

    # Calculate actual Pearson correlation coefficients
    actual_rho_mobility = np.corrcoef(mobility, pure_target)[0, 1]
    actual_rho_stringency = np.corrcoef(stringency, pure_target)[0, 1]

    # Assert that the achieved correlation is within a reasonable statistical tolerance
    # (+/- 0.15) of the requested parameter due to the added extra_noise and base signal shapes.
    assert np.isclose(actual_rho_mobility, target_correlations['mobility'], atol=0.15), \
        f"Mobility correlation {actual_rho_mobility:.2f} drifted too far from target {target_correlations['mobility']}."

    assert np.isclose(actual_rho_stringency, target_correlations['stringency'], atol=0.15), \
        f"Stringency correlation {actual_rho_stringency:.2f} drifted too far from target {target_correlations['stringency']}."


def test_generator_initial_state_validation(sir_generator: SyntheticDataGenerator):
    """
    Ensures that the generator raises a ValueError immediately if the length of the
    initial states provided does not match the dimensionality of the underlying ODE model.

    :param SyntheticDataGenerator sir_generator: The SIR generator fixture (d_s=3).
    """
    # Providing 4 states to a 3-compartment SIR model
    invalid_initial_states = [0.9, 0.1, 0.0, 0.0]
    valid_params = [0.3, 0.1]

    with pytest.raises(ValueError, match="Initial states length"):
        sir_generator.generate(initial_states=invalid_initial_states, true_params=valid_params)
