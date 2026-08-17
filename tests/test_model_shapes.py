import pytest
import torch

from einn.model.feature_module import FeatureModule
from einn.model.time_module import TimeModule


def test_feature_module_output_shape():
    """
    Tests if the FeatureModule returns a tensor of the correct shape.
    Expected output shape: [Batch, Seq_len, dim_out]
    """
    # Setting test parameters
    batch_size = 2
    seq_len = 50
    dim_seq_in = 10
    rnn_out = 40
    dim_out = 25  # A number that differs from rnn_out

    # Generating input tensors
    # FeatureModule expects three inputs: x, t, and mask optionally
    x_input = torch.rand(size=(batch_size, seq_len, dim_seq_in))
    t_input = torch.rand(size=(batch_size, seq_len, 1))

    mask_input = torch.ones(size=(batch_size, seq_len))

    # Instance creation
    model = FeatureModule(
        dim_seq_in=dim_seq_in,
        rnn_out=rnn_out,
        dim_out=dim_out,
        n_layers=1,
        bidirectional=True,
        dropout=0.0
    )

    # Just in case
    model.eval()

    # Run without calculating gradient
    with torch.no_grad():
        output = model(x=x_input, t=t_input, mask=mask_input)

    # Verifying dimensions
    # Outputs should be [Batch, Seq_len, dim_out]
    expected_shape = (batch_size, seq_len, dim_out)

    assert output.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, but got {output.shape}"


def test_time_module_output_shape():
    """
    Tests if the TimeModule returns a tensor of the correct shape.
    Expected output shape: [Batch, Seq_len, out_dim]
    """
    # Setting test parameters
    batch_size = 2
    seq_len = 50
    mapping_size = 20
    out_dim = 25  # A number that differs from mapping_size

    # Generating input tensor
    # TimeModule expects t in [Batch, Seq_len, 1]
    t_input = torch.rand(batch_size, seq_len, 1)

    # Instance creation
    model = TimeModule(
        mapping_size=mapping_size,
        scale=1.0,
        out_dim=out_dim,
        seed=42
    )
    # Just in case
    model.eval()

    # Run without calculating gradient
    with torch.no_grad():
        output = model(t=t_input)

    # Verifying dimensions
    # Outputs should be [Batch, Seq_len, out_dim]
    expected_shape = (batch_size, seq_len, out_dim)
    assert output.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, but got {output.shape}"
