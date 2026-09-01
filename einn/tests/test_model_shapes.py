import pytest
import torch

from einn.model.time_module import TimeModule


@pytest.fixture
def time_module() -> TimeModule:
    """
    Fixture that initializes and returns a TimeModule in evaluation mode.

    :return TimeModule: The initialized TimeModule in evaluation mode.
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
def base_tensor_shapes() -> dict:
    """
    Fixture providing common tensor dimensions for shape testing.

    :return dict: Dictionary containing common tensor dimensions (batch_size, seq_len, etc.).
    """
    return {
        "batch_size": 2,
        "seq_len": 50,
        "dim_seq_in": 10,
        "out_dim": 25
    }


def test_time_module_output_shape(time_module: TimeModule, base_tensor_shapes: dict):
    """
    Tests if the TimeModule returns a tensor of the correct shape.
    Expected output shape: [Batch, Seq_len, out_dim]

    :param TimeModule time_module: Fixture providing the initialized TimeModule.
    :param dict base_tensor_shapes: Fixture providing common tensor dimensions.
    """
    # Generating input tensor (t)
    t_input = torch.rand(
        size=(base_tensor_shapes["batch_size"], base_tensor_shapes["seq_len"], 1)
    )

    # Run without calculating the gradient
    with torch.no_grad():
        output = time_module(t=t_input)

    expected_shape = (
        base_tensor_shapes["batch_size"],
        base_tensor_shapes["seq_len"],
        base_tensor_shapes["out_dim"]
    )

    assert output.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, but got {output.shape}"
