import pytest
import torch

from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel
from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule


@pytest.fixture(params=["SIR", "SEIRM"])
def ode_context(request: pytest.FixtureRequest) -> dict:
    """
    Parameterized fixture providing both ODE models dynamically.
    Returns a dictionary with the initialized model, its name, the number of states (d_s),
    the number of parameters (d_p), and the expected initial parameter values.

    :param pytest.FixtureRequest request: PyTest request object for parameterization.
    :return dict: Dictionary containing model metadata and the initialized instance.
    """
    if request.param == "SIR":
        model = SIRModel()
        expected = {"beta": 0.3, "gamma": 0.1}
        model.init_params(param_dict=expected)
        return {"name": "SIR", "model": model, "d_s": 3, "d_p": 2, "expected": expected}
    else:
        model = SEIRMModel(population_n=1.0)
        expected = {"beta": 0.3, "alpha": 0.2, "gamma": 0.1, "mu": 0.05}
        model.init_params(param_dict=expected)
        return {"name": "SEIRM", "model": model, "d_s": 5, "d_p": 4, "expected": expected}


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
def output_module(ode_context: dict) -> OutputModule:
    """
    Fixture initializing the OutputModule.
    Dynamically sets the output dimension (d_s) based on the current ODE model in the context.

    :param dict ode_context: Fixture providing the current ODE model metadata.
    """
    model = OutputModule(
        d_e=25,
        d_s=ode_context["d_s"]
    )
    model.eval()
    return model


@pytest.fixture
def mock_inputs() -> dict:
    """
    Fixture providing batch inputs for the forward pass.
    """
    return {
        "t": torch.rand(size=(2, 50, 1))  # Batch=2, Seq_len=50, Features=1
    }


def test_ode_parameter_initialization(ode_context: dict):
    """
    Tests if all specific parameters for the given model (SIR or SEIRM)
    are properly initialized and scaled to their expected values.

    :param dict ode_context: Fixture providing the ODE model context.
    """
    model = ode_context["model"]
    expected_params = ode_context["expected"]
    d_p = ode_context["d_p"]

    scaled_params = model.get_scaled_params(detach=True)

    assert scaled_params.shape[0] == d_p, \
        f"{ode_context['name']} expected {d_p} parameters, got {scaled_params.shape[0]}."

    for i, (param_name, expected_val) in enumerate(expected_params.items()):
        assert torch.allclose(
            input=scaled_params[i],
            other=torch.tensor(data=expected_val, dtype=torch.float32),
            atol=1e-4
        ), f"{ode_context['name']} parameter '{param_name}' mismatch!"


def test_ode_parameter_bounding_extreme_optimizer(ode_context: dict):
    """
    Simulates a scenario where the optimizer updates the raw parameters to extreme values.
    Ensures the tanh trick successfully bounds all physical parameters exactly between 0 and 1.

    :param dict ode_context: Fixture providing the ODE model context.
    """
    model = ode_context["model"]
    d_p = ode_context["d_p"]

    # Manually setting extreme raw values to simulate optimizer overshoot for all params dynamically
    with torch.no_grad():
        for i in range(d_p):
            if i % 2 == 0:
                model.raw_params[i] = 50.0  # Extremely high -> should bound to 1.0
            else:
                model.raw_params[i] = -50.0  # Extremely low -> should bound to 0.0

    scaled_params = model.get_scaled_params(detach=True)

    for i in range(d_p):
        expected_val = 1.0 if i % 2 == 0 else 0.0
        assert torch.allclose(
            input=scaled_params[i], other=torch.tensor(data=expected_val, dtype=torch.float32)
        ), f"{ode_context['name']}: Parameter {i} did not properly bound to {expected_val}."


def test_neural_to_ode_full_pipeline(
        time_module: TimeModule,
        output_module: OutputModule,
        ode_context: dict,
        mock_inputs: dict
):
    """
    Tests the actual EINN use case:
    1. Neural networks predict latent states.
    2. OutputModule decodes them into physical compartments (S, I, R or S, E, I, R, M).
    3. The ODE model calculates the time derivatives from these predicted compartments.

    :param TimeModule time_module: Fixture providing the TimeModule.
    :param OutputModule output_module: Fixture providing the OutputModule.
    :param dict ode_context: Fixture providing the ODE model context.
    :param dict mock_inputs: Fixture providing input tensors.
    """
    model = ode_context["model"]
    d_s = ode_context["d_s"]

    with torch.no_grad():
        # 1. Generate latent embedding using TimeModule
        e_t = time_module(t=mock_inputs["t"])

        # 2. Decode into raw physical states. Shape: [Batch, Seq, d_s]
        predicted_states = output_module(e=e_t)

        # To ensure the states act like physical probabilities/fractions (0 to 1),
        # we apply a sigmoid activation. (In the real trainer, softmax/sigmoid handles this).
        normalized_states = torch.sigmoid(input=predicted_states)

        # 3. Calculate ODE derivatives based on network predictions
        ode_solution = model.get_derivatives(
            states=normalized_states,
            detach_params=False
        )

    expected_shape = (2, 50, d_s)

    # Check if the mathematical flow preserved the correct tensor shapes
    assert ode_solution.ds_dt.shape == expected_shape, \
        f"{ode_context['name']} derivative shape mismatch! Expected {expected_shape}, got {ode_solution.ds_dt.shape}"

    # Physics check: In a closed system, the sum of derivatives MUST equal 0
    ds_dt_sum = ode_solution.ds_dt.sum(dim=-1)
    expected_zeros = torch.zeros_like(input=ds_dt_sum)

    assert torch.allclose(input=ds_dt_sum, other=expected_zeros, atol=1e-5), \
        f"{ode_context['name']}: The closed system of equations is violated if the sum of derivatives is not 0."
