import pytest
import torch

from einn.model.feature_module import FeatureModule
from einn.model.output_module import OutputModule
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
def feature_module() -> FeatureModule:
    """
    Fixture that initializes and returns a FeatureModule in evaluation mode.

    :return FeatureModule: The initialized FeatureModule in evaluation mode.
    """
    model = FeatureModule(
        dim_seq_in=10,
        rnn_out=40,
        dim_out=25,
        n_layers=1,
        bidirectional=True,
        dropout=0.0
    )
    model.eval()
    return model


@pytest.fixture
def output_module(base_tensor_shapes: dict) -> OutputModule:
    """
    Fixture that initializes and returns an OutputModule in evaluation mode.
    The input dimension (d_e) matches the concatenated outputs of the upstream modules.

    :param dict base_tensor_shapes: Fixture providing common tensor dimensions.
    :return OutputModule: The initialized OutputModule in evaluation mode.
    """
    model = OutputModule(
        d_e=base_tensor_shapes["d_e"],
        d_s=base_tensor_shapes["d_s"]
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
        "out_dim": 25,
        "d_e": 25,  # Same as out_dim
        "d_s": 5
    }


@pytest.fixture
def input_tensors(base_tensor_shapes: dict) -> dict:
    """
    Fixture generating common input tensors (x, t, mask) to avoid code duplication.

    :param dict base_tensor_shapes: Fixture providing common tensor dimensions.
    :return dict: Dictionary containing the generated PyTorch tensors.
    """
    batch_size = base_tensor_shapes["batch_size"]
    seq_len = base_tensor_shapes["seq_len"]

    return {
        "x": torch.rand(size=(batch_size, seq_len, base_tensor_shapes["dim_seq_in"])),
        "t": torch.rand(size=(batch_size, seq_len, 1)),
        "mask": torch.ones(size=(batch_size, seq_len))
    }


def test_time_module_output_shape(time_module: TimeModule, base_tensor_shapes: dict, input_tensors: dict):
    """
    Tests if the TimeModule returns a tensor of the correct shape.
    Expected output shape: [Batch, Seq_len, out_dim]

    :param TimeModule time_module: Fixture providing the initialized TimeModule.
    :param dict base_tensor_shapes: Fixture providing common tensor dimensions.
    :param dict input_tensors: Fixture providing the input tensors.
    """
    with torch.no_grad():
        output = time_module(t=input_tensors["t"])

    expected_shape = (
        base_tensor_shapes["batch_size"],
        base_tensor_shapes["seq_len"],
        base_tensor_shapes["out_dim"]
    )

    assert output.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, but got {output.shape}"


def test_feature_module_output_shape(feature_module: FeatureModule, base_tensor_shapes: dict, input_tensors: dict):
    """
    Tests if the FeatureModule returns a tensor of the correct shape.
    Expected output shape: [Batch, Seq_len, dim_out]

    :param FeatureModule feature_module: Fixture providing the initialized FeatureModule.
    :param dict base_tensor_shapes: Fixture providing common tensor dimensions.
    :param dict input_tensors: Fixture providing the input tensors.
    """
    with torch.no_grad():
        output = feature_module(
            x=input_tensors["x"],
            t=input_tensors["t"],
            mask=input_tensors["mask"]
        )

    expected_shape = (
        base_tensor_shapes["batch_size"],
        base_tensor_shapes["seq_len"],
        base_tensor_shapes["out_dim"]
    )

    assert output.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, but got {output.shape}"


def test_combined_modules_compatibility(
        time_module: TimeModule,
        feature_module: FeatureModule,
        base_tensor_shapes: dict,
        input_tensors: dict
):
    """
    Tests the compatibility of TimeModule and FeatureModule outputs.
    Ensures both modules can process the same inputs and their outputs can be combined.

    :param TimeModule time_module: Fixture providing the initialized TimeModule.
    :param FeatureModule feature_module: Fixture providing the initialized FeatureModule.
    :param dict base_tensor_shapes: Fixture providing common tensor dimensions.
    :param dict input_tensors: Fixture providing the input tensors.
    """
    with torch.no_grad():
        time_output = time_module(t=input_tensors["t"])
        feature_output = feature_module(
            x=input_tensors["x"],
            t=input_tensors["t"],
            mask=input_tensors["mask"]
        )

    # Concatenation
    combined_output = torch.cat(tensors=(time_output, feature_output), dim=-1)

    # Expected dimension should be the sum of the two out_dim (25 + 25)
    expected_shape = (
        base_tensor_shapes["batch_size"],
        base_tensor_shapes["seq_len"],
        base_tensor_shapes["out_dim"] * 2
    )

    assert combined_output.shape == expected_shape, \
        f"Combined shape mismatch! Expected {expected_shape}, but got {combined_output.shape}"


def test_output_module_output_shape(output_module: OutputModule, base_tensor_shapes: dict):
    """
    Tests if the OutputModule returns a tensor of the correct shape in a standalone manner.
    Expected output shape: [Batch, Seq_len, d_s]

    :param OutputModule output_module: Fixture providing the initialized OutputModule.
    :param dict base_tensor_shapes: Fixture providing common tensor dimensions.
    """
    # Generating a mock combined embedding (e) as input
    e_input = torch.rand(
        size=(base_tensor_shapes["batch_size"], base_tensor_shapes["seq_len"], base_tensor_shapes["d_e"])
    )

    with torch.no_grad():
        output = output_module(e=e_input)

    expected_shape = (
        base_tensor_shapes["batch_size"],
        base_tensor_shapes["seq_len"],
        base_tensor_shapes["d_s"]
    )

    assert output.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, but got {output.shape}"


def test_full_pipeline_shared_mlp(
        time_module: TimeModule,
        feature_module: FeatureModule,
        output_module: OutputModule,
        base_tensor_shapes: dict,
        input_tensors: dict
):
    """
    Tests the full pipeline where the OutputModule acts as a shared MLP.
    It must independently decode both the TimeModule's and FeatureModule's embeddings.

    :param TimeModule time_module: Fixture providing the initialized TimeModule.
    :param FeatureModule feature_module: Fixture providing the initialized FeatureModule.
    :param OutputModule output_module: Fixture providing the initialized OutputModule.
    :param dict base_tensor_shapes: Fixture providing common tensor dimensions.
    :param dict input_tensors: Fixture providing the raw input tensors.
    """
    with torch.no_grad():
        time_embedding = time_module(t=input_tensors["t"])
        feature_embedding = feature_module(
            x=input_tensors["x"],
            t=input_tensors["t"],
            mask=input_tensors["mask"]
        )

        states_from_time = output_module(e=time_embedding)
        states_from_feature = output_module(e=feature_embedding)

    # Final expected dimension [Batch, Seq_len, d_s]
    expected_shape = (
        base_tensor_shapes["batch_size"],
        base_tensor_shapes["seq_len"],
        base_tensor_shapes["d_s"]
    )

    assert states_from_time.shape == expected_shape, \
        f"Time pipeline shape mismatch! Expected {expected_shape}, but got {states_from_time.shape}"

    assert states_from_feature.shape == expected_shape, \
        f"Feature pipeline shape mismatch! Expected {expected_shape}, but got {states_from_feature.shape}"
