import pytest
import torch

from einn.model.time_module import TimeModule


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
