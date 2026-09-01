import pytest
import torch

from einn.data.einn_dataset import EINNDataset


@pytest.fixture
def dataset_dims() -> dict:
    """
    Fixture providing common sequence dimensions, feature sizes, and window sizes.

    :return dict: Dictionary containing dataset dimensions and window size configuration.
    """
    return {
        "batch_size": 2,
        "seq_len": 100,
        "x_feat": 10,
        "y_feat": 1,
        "t_feat": 1,
        "aux_feat": 5,
        "window_size": 21
    }


@pytest.fixture
def mock_2d_tensors(dataset_dims: dict) -> dict:
    """
    Fixture generating 2D tensors [Seq_len, Features] for basic dataset testing.

    :param dict dataset_dims: Fixture providing dataset dimension configurations.
    :return dict: Dictionary of generated 2D mock tensors for x, y, t, and aux_targets.
    """
    seq_len = dataset_dims["seq_len"]
    return {
        "x": torch.rand(size=(seq_len, dataset_dims["x_feat"])),
        "y": torch.rand(size=(seq_len, dataset_dims["y_feat"])),
        "t": torch.rand(size=(seq_len, dataset_dims["t_feat"])),
        "aux_targets": torch.rand(size=(seq_len, dataset_dims["aux_feat"]))
    }


@pytest.fixture
def mock_3d_tensors(dataset_dims: dict) -> dict:
    """
    Fixture generating 3D tensors [Batch, Seq_len, Features] for edge-case testing.

    :param dict dataset_dims: Fixture providing dataset dimension configurations.
    :return dict: Dictionary of generated 3D mock tensors for x, y, t, and aux_targets.
    """
    batch = dataset_dims["batch_size"]
    seq_len = dataset_dims["seq_len"]
    return {
        "x": torch.rand(size=(batch, seq_len, dataset_dims["x_feat"])),
        "y": torch.rand(size=(batch, seq_len, dataset_dims["y_feat"])),
        "t": torch.rand(size=(batch, seq_len, dataset_dims["t_feat"])),
        "aux_targets": torch.rand(size=(batch, seq_len, dataset_dims["aux_feat"]))
    }


def test_dataset_full_sequence(mock_2d_tensors: dict, dataset_dims: dict):
    """
    Tests dataset when it receives a full sequence as one element (window).

    :param dict mock_2d_tensors: Fixture providing generated 2D tensors.
    :param dict dataset_dims: Fixture providing dataset dimension configurations.
    """
    dataset = EINNDataset(
        x=mock_2d_tensors["x"],
        y=mock_2d_tensors["y"],
        t=mock_2d_tensors["t"],
        aux_targets=mock_2d_tensors["aux_targets"],
        window_size=None
    )

    # Verifying that length of the dataset is 1
    assert len(dataset) == 1

    # Verifying that the elements detached are the expected size
    item = dataset[0]
    assert isinstance(item, dict)
    assert item['X'].shape == (dataset_dims["seq_len"], dataset_dims["x_feat"])
    assert item['y'].shape == (dataset_dims["seq_len"], dataset_dims["y_feat"])
    assert item['t'].shape == (dataset_dims["seq_len"], dataset_dims["t_feat"])
    assert item['aux_targets'].shape == (dataset_dims["seq_len"], dataset_dims["aux_feat"])


def test_dataset_sliding_window(mock_2d_tensors: dict, dataset_dims: dict):
    """
    Tests the sliding windows method (batch size > 1).

    :param dict mock_2d_tensors: Fixture providing generated 2D tensors.
    :param dict dataset_dims: Fixture providing dataset dimension configurations.
    """
    window = dataset_dims["window_size"]

    dataset = EINNDataset(
        x=mock_2d_tensors["x"],
        y=mock_2d_tensors["y"],
        t=mock_2d_tensors["t"],
        aux_targets=mock_2d_tensors["aux_targets"],
        window_size=window
    )

    # Length should be (seq_len - window_size + 1)
    expected_length = dataset_dims["seq_len"] - window + 1
    assert len(dataset) == expected_length

    # Verifying the detached elements' sizes are as expected (window size)
    item = dataset[5]  # Randomly chosen index
    assert item['X'].shape == (window, dataset_dims["x_feat"])
    assert item['y'].shape == (window, dataset_dims["y_feat"])
    assert item['t'].shape == (window, dataset_dims["t_feat"])
    assert item['aux_targets'].shape == (window, dataset_dims["aux_feat"])


def test_dataset_3d_tensor_size_not_one(mock_3d_tensors: dict, dataset_dims: dict):
    """
    Tests dataset when the input is a 3D tensor but size(0) != 1. Covers the edge case where
    the squeeze condition evaluates to False.

    :param dict mock_3d_tensors: Fixture providing generated 3D tensors.
    :param dict dataset_dims: Fixture providing dataset dimension configurations.
    """
    dataset = EINNDataset(
        x=mock_3d_tensors["x"],
        y=mock_3d_tensors["y"],
        t=mock_3d_tensors["t"],
        aux_targets=mock_3d_tensors["aux_targets"],
        window_size=None
    )

    # Current behavior: Since no squeeze occurred, the dataset treats batch_size as the
    # sequence length. (len = batch_size - batch_size + 1 = 1)
    assert len(dataset) == 1

    # Verifying the detached elements' sizes are as expected
    item = dataset[0]
    batch = dataset_dims["batch_size"]
    seq = dataset_dims["seq_len"]

    assert isinstance(item, dict)
    assert item['X'].shape == (batch, seq, dataset_dims["x_feat"])
    assert item['y'].shape == (batch, seq, dataset_dims["y_feat"])
    assert item['t'].shape == (batch, seq, dataset_dims["t_feat"])
    assert item['aux_targets'].shape == (batch, seq, dataset_dims["aux_feat"])
