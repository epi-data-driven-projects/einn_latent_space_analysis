import pytest
import torch

from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel
from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule


@pytest.fixture(params=["SIR", "SEIRM"])
def ode_context(request: pytest.FixtureRequest) -> dict:
    """
    Parameterized fixture providing both SIR and SEIRM models dynamically.
    Returns a dictionary with the initialized model, its name, and the number of states (d_s).

    :param pytest.FixtureRequest request: PyTest request object for parameterization.
    :return dict: Dictionary containing model metadata and the initialized instance.
    """
    if request.param == "SIR":
        model = SIRModel()
        model.init_params(param_dict={"beta": 0.3, "gamma": 0.1})
        return {"name": "SIR", "model": model, "d_s": 3}
    else:
        model = SEIRMModel(population_n=1.0)
        model.init_params(param_dict={"beta": 0.3, "alpha": 0.2, "gamma": 0.1, "mu": 0.05})
        return {"name": "SEIRM", "model": model, "d_s": 5}


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


def test_ode_parameter_bounding_extreme_optimizer(ode_context: dict):
    """
    Simulates a scenario where the optimizer updates the raw parameters to extreme values.
    Ensures the tanh trick successfully bounds the physical parameters exactly between 0 and 1.

    :param dict ode_context: Fixture providing the ODE model context.
    """
    model = ode_context["model"]

    # Manually setting extreme raw values to simulate optimizer overshoot for the first two params
    with torch.no_grad():
        model.raw_params[0] = 50.0   # Extremely high
        model.raw_params[1] = -50.0  # Extremely low

    scaled_params = model.get_scaled_params(detach=True)

    assert torch.allclose(
        input=scaled_params[0], other=torch.tensor(data=1.0, dtype=torch.float32)
    ), f"{ode_context['name']}: Parameter 0 did not properly bound to the maximum limit (1.0)."

    assert torch.allclose(
        input=scaled_params[1], other=torch.tensor(data=0.0, dtype=torch.float32)
    ), f"{ode_context['name']}: Parameter 1 did not properly bound to the minimum limit (0.0)."


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
