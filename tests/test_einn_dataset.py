import pytest
import torch

from einn.data.einn_dataset import EINNDataset


def test_dataset_full_sequence():
    """
    Tests dataset, when it receives a full sequence as one element (window).
    """
    seq_len = 100

    # Generating random tensors
    x = torch.rand(seq_len, 10)
    y = torch.rand(seq_len, 1)
    t = torch.rand(seq_len, 1)
    aux_targets = torch.rand(seq_len, 5)

    dataset = EINNDataset(x=x, y=y, t=t, aux_targets=aux_targets, window_size=None)

    # Verifying that length of the dataset is 1
    assert len(dataset) == 1

    # Verifying that the elements detached are the expected size
    item = dataset[0]
    assert isinstance(item, dict)
    assert item['X'].shape == (100, 10)
    assert item['y'].shape == (100, 1)
    assert item['t'].shape == (100, 1)
    assert item['aux_targets'].shape == (100, 5)


def test_dataset_sliding_window():
    """
    Tests the sliding windows method (batch size > 1)
    """
    seq_len = 100
    window_size = 21

    x = torch.rand(seq_len, 10)
    y = torch.rand(seq_len, 1)
    t = torch.rand(seq_len, 1)
    aux_targets = torch.rand(seq_len, 5)

    dataset = EINNDataset(x=x, y=y, t=t, aux_targets=aux_targets, window_size=window_size)

    # Length should be (seq_len - window_size + 1)
    expected_length = 100 - 21 + 1
    assert len(dataset) == expected_length

    # Verifying the detached elements' sizes are as expected (window size)
    item = dataset[5]  # Randomly chosen index
    assert item['X'].shape == (21, 10)
    assert item['y'].shape == (21, 1)
    assert item['t'].shape == (21, 1)
    assert item['aux_targets'].shape == (21, 5)


def test_dataset_3d_tensor_size_not_one():
    """
    Tests dataset when the input is a 3D tensor but size(0) != 1. Covers the edge case where
     the squeeze condition evaluates to False.
    """
    batch_size = 2
    seq_len = 100

    # Generating 3D tensors where size(0) = 2
    x = torch.rand(batch_size, seq_len, 10)
    y = torch.rand(batch_size, seq_len, 1)
    t = torch.rand(batch_size, seq_len, 1)
    aux_targets = torch.rand(batch_size, seq_len, 5)

    dataset = EINNDataset(x=x, y=y, t=t, aux_targets=aux_targets, window_size=None)

    # Current behavior: Since no squeeze occurred, the dataset treats 2 (batch_size) as the
    # sequence length. (len = 2 - 2 + 1 = 1)
    assert len(dataset) == 1

    # Verifying the detached elements' sizes are as expected
    item = dataset[0]
    assert isinstance(item, dict)
    assert item['X'].shape == (2, 100, 10)
    assert item['y'].shape == (2, 100, 1)
    assert item['t'].shape == (2, 100, 1)
    assert item['aux_targets'].shape == (2, 100, 5)
