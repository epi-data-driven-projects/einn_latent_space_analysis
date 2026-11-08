import pytest

import numpy as np
import torch

from einn.data.ode_solution_generator import ODESolutionGenerator
from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel


@pytest.fixture
def sir_generator() -> ODESolutionGenerator:
    """
    Fixture providing an ODESolutionGenerator initialized with the SIR model.

    :return ODESolutionGenerator: Initialized calibrator instance (d_p = 2).
    """
    model = SIRModel(population_n=1.0, target_idx=1)
    return ODESolutionGenerator(assumed_ode_model=model, d_p=2, device='cpu')


@pytest.fixture
def seirm_generator() -> ODESolutionGenerator:
    """
    Fixture providing an ODESolutionGenerator initialized with the SEIRM model.

    :return ODESolutionGenerator: Initialized calibrator instance (d_p = 4).
    """
    model = SEIRMModel(population_n=1.0, target_idx=4)
    return ODESolutionGenerator(assumed_ode_model=model, d_p=4, device='cpu')


def test_calibrator_output_shape_and_type(sir_generator: ODESolutionGenerator):
    """
    Verifies that the calibration process yields a properly formatted NumPy array
    matching the expected sequence length and mathematical compartment dimensions (d_s).

    :param ODESolutionGenerator sir_generator: The SIR generator fixture.
    """
    seq_len = 50
    dummy_y_raw = np.random.rand(seq_len, 1).astype(np.float32)
    initial_states = [0.99, 0.01, 0.0]

    calibrated_curves = sir_generator.calibrate(
        y_raw=dummy_y_raw,
        initial_states=initial_states,
        epochs=2  # Keep epochs low for simple shape testing
    )

    assert isinstance(calibrated_curves, np.ndarray), "Generator must return a NumPy array."
    # Expecting [Seq_len, d_s] -> [50, 3] for SIR
    assert calibrated_curves.shape == (seq_len, 3), \
        f"Expected output shape {(seq_len, 3)}, got {calibrated_curves.shape}."


def test_initial_state_anchoring(seirm_generator: ODESolutionGenerator):
    """
    Validates that the very first row of the generated latent trajectory
    strictly matches the provided initial biological conditions (t=0).
    The optimizer is not allowed to mutate the starting state.

    :param ODESolutionGenerator seirm_generator: The SEIRM generator fixture.
    """
    seq_len = 30
    dummy_y_raw = np.zeros((seq_len, 1), dtype=np.float32)

    # S, E, I, R, M
    locked_initial_states = [0.95, 0.02, 0.03, 0.0, 0.0]

    calibrated_curves = seirm_generator.calibrate(
        y_raw=dummy_y_raw,
        initial_states=locked_initial_states,
        epochs=1
    )

    # Extract the first row (t=0)
    t0_states = calibrated_curves[0].tolist()

    assert np.allclose(t0_states, locked_initial_states, atol=1e-5), \
        f"Initial state anchoring failed. Expected {locked_initial_states}, but the trajectory started at {t0_states}."


def test_lbfgs_convergence_accuracy(sir_generator: ODESolutionGenerator):
    """
    Advanced mathematical integration test:
    Generates a pure, deterministic SIR trajectory with known 'true' parameters.
    Then, initializes the generator with deliberately 'wrong' initial guesses to verify
    that the L-BFGS optimizer successfully minimizes the error and recovers the curve.

    :param ODESolutionGenerator sir_generator: The SIR generator fixture.
    """
    seq_len = 60
    initial_states = [0.99, 0.01, 0.0]

    # 1. Generate the 'Ground Truth' target curve
    true_beta, true_gamma = 0.5, 0.1
    current_state = torch.tensor([initial_states], dtype=torch.float32)
    true_params = torch.tensor([[true_beta, true_gamma]], dtype=torch.float32)

    y_raw = []

    # Forward Euler simulation matching the calibrator's internal logic
    for _ in range(seq_len):
        y_raw.append(current_state[0, 1].item())  # 1 is target_idx (Infected)
        derivatives = sir_generator.model(states=current_state, params=true_params)
        current_state = current_state + derivatives * 1.0

    y_raw_array = np.array(y_raw, dtype=np.float32).reshape(-1, 1)

    # 2. Run the generator starting from a mathematically recoverable, but wrong guess.
    bad_initial_guess = [0.3, 0.2]

    calibrated_curves = sir_generator.calibrate(
        y_raw=y_raw_array,
        initial_states=initial_states,
        initial_guess=bad_initial_guess,
        epochs=30,
        lr=0.1
    )

    # Extract the calibrator's final simulated Infected curve
    predicted_y = calibrated_curves[:, sir_generator.target_idx].reshape(-1, 1)

    # 3. Validation: The Mean Squared Error between the pure target and the
    # calibrated prediction should be extremely small, proving convergence.
    mse_error = np.mean((y_raw_array - predicted_y) ** 2)

    assert mse_error < 1e-4, \
        f"L-BFGS optimization failed to converge. MSE Error was {mse_error:.6f}, which is above the 1e-4 tolerance."


def test_invalid_initial_guess_length(sir_generator: ODESolutionGenerator):
    """
    Ensures the generator safely raises a ValueError if the provided initial parameter
    guess array does not match the dimensionality of the underlying ODE physics (d_p).

    :param ODESolutionGenerator sir_generator: The SIR generator fixture (d_p=2).
    """
    seq_len = 10
    dummy_y_raw = np.random.rand(seq_len, 1).astype(np.float32)
    initial_states = [0.99, 0.01, 0.0]

    # SIR expects 2 parameters, we provide 3
    invalid_guess = [0.3, 0.2, 0.1]

    with pytest.raises(ValueError, match="must match d_p"):
        sir_generator.calibrate(
            y_raw=dummy_y_raw,
            initial_states=initial_states,
            initial_guess=invalid_guess
        )
