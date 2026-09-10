import pytest
import torch

from einn.config.einn_config import EINNConfig
from einn.loss.einn_loss import EINNLoss
from einn.model.interface.phase_context import PhaseContext


@pytest.fixture
def base_config() -> EINNConfig:
    """
    Fixture providing a base configuration on CPU.
    """
    return EINNConfig(device='cpu')


@pytest.fixture
def loss_calculator_seirm(base_config: EINNConfig) -> EINNLoss:
    """
    Fixture providing an initialized EINNLoss for the SEIRM model.
    """
    return EINNLoss(config=base_config, ode_model=None, model_type="SEIRM")


def test_monotonicity_loss_penalties(loss_calculator_seirm: EINNLoss) -> None:
    """
    Tests if the asymmetric squared ReLU correctly penalizes violating derivatives.
    For SEIRM: S (idx 0) must decrease, R (idx 3) and M (idx 4) must increase.
    """
    # Test case: S is improperly increasing (+2.0), R is improperly decreasing (-3.0), M is valid (0.0)
    dS_dt = torch.tensor(data=[[[2.0, 0.0, 0.0, -3.0, 0.0]]], dtype=torch.float32)

    mono_loss = loss_calculator_seirm.calc_monotonicity_loss(
        dS_dt=dS_dt,
        inc_indices=loss_calculator_seirm.mono_inc_indices,
        dec_indices=loss_calculator_seirm.mono_dec_indices
    )

    # S penalty: (2.0 * relu(2.0))^2 = 16.0
    # R penalty: (-(-3.0) * relu(3.0))^2 = 81.0
    # Total expected: 97.0
    expected_loss = torch.tensor(data=97.0, dtype=torch.float32)

    assert torch.allclose(input=mono_loss, other=expected_loss, atol=1e-4), \
        "Monotonicity penalty calculated incorrectly."


def test_parameter_smoothness_loss(loss_calculator_seirm: EINNLoss) -> None:
    """
    Tests if the smoothness loss accurately calculates the squared differences between consecutive steps.
    """
    params = torch.tensor(data=[[[0.1], [0.5], [0.3]]], dtype=torch.float32)

    # Diff 1: 0.5 - 0.1 = 0.4 -> squared = 0.16
    # Diff 2: 0.3 - 0.5 = -0.2 -> squared = 0.04
    # Mean of (0.16, 0.04) = 0.10
    expected_loss = torch.tensor(data=0.10, dtype=torch.float32)

    smooth_loss = loss_calculator_seirm.calc_parameter_smoothness_loss(params=params)
    assert torch.allclose(input=smooth_loss, other=expected_loss, atol=1e-4), \
        "Smoothness loss calculated incorrectly."


def test_forward_phase_routing(loss_calculator_seirm: EINNLoss) -> None:
    """
    Tests if the cascading forward pass correctly aggregates active losses without KeyErrors
    when executing the final phase (Phase 4).
    """
    mock_tensor = torch.zeros(size=(1, 5, 5))
    mock_params = torch.zeros(size=(1, 5, 4))

    network_outputs = {
        's_t': mock_tensor,
        's_t_F': mock_tensor,
        'e_t': mock_tensor,
        'e_t_F': mock_tensor,
        'ds_dt_T_nn': mock_tensor,
        'ds_dt_T_ode': mock_tensor,
        'ds_dt_future_T_nn': mock_tensor,
        'ds_dt_future_T_ode': mock_tensor,
        'dS_dt_T_ode': mock_tensor,
        'ds_dt_F_nn': mock_tensor,
        'ds_dt_F_ode': mock_tensor,
        'ds_dt_future_F_nn': mock_tensor,
        'ds_dt_future_F_ode': mock_tensor,
        'params': mock_params
    }

    context = PhaseContext(
        phase_num=4,
        epoch=1,
        X=mock_tensor,
        y=mock_tensor,
        t=torch.zeros(size=(1, 5, 1)),
        aux_targets=mock_tensor,
        models=None
    )

    try:
        loss = loss_calculator_seirm(phase_context=context, network_outputs=network_outputs)
        assert isinstance(loss, torch.Tensor), "Forward pass must return a Tensor."
        assert not torch.isnan(input=loss), "Loss computation resulted in NaN."
    except KeyError as e:
        pytest.fail(reason=f"Phase routing failed due to missing key in network_outputs: {e}")
