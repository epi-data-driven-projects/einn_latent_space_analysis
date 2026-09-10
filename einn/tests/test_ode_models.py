import pytest
import torch

from einn.ode.sir_model import SIRModel
from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule


@pytest.fixture
def time_module() -> TimeModule:
    """
    Fixture initializing the TimeModule.
    """
    model = TimeModule(
        mapping_size=20,
        scale=1.0,
        out_dim=25,
        seed=42
    )
    model.eval()
    return model


@pytest.fixture
def output_module() -> OutputModule:
    """
    Fixture initializing the OutputModule.
    Set to output 3 states (S, I, R) to match the SIR model.
    """
    model = OutputModule(
        d_e=25,
        d_s=3
    )
    model.eval()
    return model


@pytest.fixture
def sir_model() -> SIRModel:
    """
    Fixture initializing the concrete SIR ODE Model.
    """
    model = SIRModel()
    model.init_params(param_dict={"beta": 0.3, "gamma": 0.1})
    return model


@pytest.fixture
def mock_inputs() -> dict:
    """
    Fixture providing batch inputs for the forward pass.
    """
    return {
        "t": torch.rand(size=(2, 50, 1))  # Batch=2, Seq_len=50, Features=1
    }


def test_ode_parameter_bounding_extreme_optimizer(sir_model: SIRModel):
    """
    Simulates a scenario where the optimizer updates the raw parameters to extreme values.
    Ensures the tanh trick successfully bounds the physical parameters exactly between 0 and 1.

    :param SIRModel sir_model: Fixture providing the initialized SIR model.
    """
    # Manually setting extreme raw values to simulate optimizer overshoot
    with torch.no_grad():
        sir_model.raw_params[0] = 50.0  # Extremely high raw beta
        sir_model.raw_params[1] = -50.0  # Extremely low raw gamma

    scaled_params = sir_model.get_scaled_params(detach=True)

    # Beta should cap near 1.0, gamma should floor near 0.0
    assert torch.allclose(
        input=scaled_params[0], other=torch.tensor(data=1.0, dtype=torch.float32)
    ), "Beta did not properly bound to the maximum limit (1.0)."

    assert torch.allclose(
        input=scaled_params[1], other=torch.tensor(data=0.0, dtype=torch.float32)
    ), "Gamma did not properly bound to the minimum limit (0.0)."


def test_neural_to_ode_full_pipeline(
        time_module: TimeModule,
        output_module: OutputModule,
        sir_model: SIRModel,
        mock_inputs: dict
):
    """
    Tests the actual EINN use case:
    1. Neural networks predict latent states.
    2. OutputModule decodes them into physical compartments (S, I, R).
    3. The ODE model calculates the time derivatives from these predicted compartments.

    :param TimeModule time_module: Fixture providing the TimeModule.
    :param OutputModule output_module: Fixture providing the OutputModule.
    :param SIRModel sir_model: Fixture providing the SIR model.
    :param dict mock_inputs: Fixture providing input tensors.
    """
    with torch.no_grad():
        # 1. Generate latent embedding using TimeModule
        e_t = time_module(t=mock_inputs["t"])

        # 2. Decode into raw physical SIR states. Shape: [Batch, Seq, 3]
        predicted_states = output_module(e=e_t)

        # To ensure the states act like physical probabilities/fractions (0 to 1),
        # we apply a sigmoid activation. (In the real trainer, softmax/sigmoid handles this).
        normalized_states = torch.sigmoid(input=predicted_states)

        # 3. Calculate ODE derivatives based on network predictions
        ode_solution = sir_model.get_derivatives(
            states=normalized_states,
            detach_params=False
        )

    expected_shape = (2, 50, 3)

    # Check if the mathematical flow preserved the correct tensor shapes
    assert ode_solution.ds_dt.shape == expected_shape, \
        f"Derivative shape mismatch! Expected {expected_shape}, got {ode_solution.ds_dt.shape}"

    # Physics check: In a closed SIR system, the sum of derivatives (dS/dt + dI/dt + dR/dt) MUST equal 0
    ds_dt_sum = ode_solution.ds_dt.sum(dim=-1)
    expected_zeros = torch.zeros_like(input=ds_dt_sum)

    assert torch.allclose(input=ds_dt_sum, other=expected_zeros, atol=1e-5), \
        "The closed system of equations is violated if the sum of derivatives is not 0."
