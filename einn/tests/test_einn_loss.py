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


def test_monotonicity_loss_penalties(loss_calculator_seirm: EINNLoss):
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


def test_parameter_smoothness_loss(loss_calculator_seirm: EINNLoss):
    """
    Tests if the smoothness loss accurately calculates the squared differences between consecutive steps.
    """
    params = torch.tensor(data=[[[0.1], [0.5], [0.3]]], dtype=torch.float32)

    # Diff 1: 0.5 - 0.1 = 0.4 -> squared = 0.16
    # Diff 2: 0.3 - 0.5 = -0.2 -> squared = 0.04
    # Mean of (0.16, 0.04) = 0.10
    expected_loss = torch.tensor(data=0.10, dtype=torch.float32)

    smooth_loss = loss_calculator_seirm.calc_parameter_smoothness_loss(params=params)
    assert torch.allclose(input=smooth_loss, other=expected_loss, atol=1e-5), \
        "Smoothness loss calculated incorrectly."


def test_forward_phase_routing(loss_calculator_seirm: EINNLoss):
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


def test_phase_routing_and_weight_integration(base_config: EINNConfig):
    """
    Advanced combined test: Verifies that loss weights are correctly applied and
    that the cascading phase logic perfectly accumulates the expected losses.
    """
    base_config.loss_weights = {
        'data_T': 2.0, 'aux': 3.0, 'mono': 0.0, 'ode_T': 0.0, 'ode_future_T': 0.0,
        'param': 0.0, 'data_F': 0.0, 'kd_target': 0.0, 'kd_emb': 0.0, 'ode_F': 0.0, 'ode_future_F': 0.0
    }
    loss_calculator = EINNLoss(config=base_config, ode_model=None, model_type="SIR")

    # MSE should be 1.0
    mock_targets = torch.zeros(size=(1, 5, 1))
    mock_states = torch.ones(size=(1, 5, 3))

    network_outputs = {
        's_t': mock_states, 's_t_F': mock_states, 'e_t': mock_states, 'e_t_F': mock_states,
        'ds_dt_T_nn': mock_states, 'ds_dt_T_ode': mock_states, 'ds_dt_future_T_nn': mock_states,
        'ds_dt_future_T_ode': mock_states, 'dS_dt_T_ode': mock_states, 'ds_dt_F_nn': mock_states,
        'ds_dt_F_ode': mock_states, 'ds_dt_future_F_nn': mock_states, 'ds_dt_future_F_ode': mock_states,
        'params': torch.ones(size=(1, 5, 2))
    }

    context_phase1 = PhaseContext(
        phase_num=1, epoch=1, X=mock_targets, y=mock_targets,
        t=torch.zeros(size=(1, 5, 1)), aux_targets=mock_targets, models=None
    )

    # In Phase 1: data_T (2.0) + aux (3.0) -> Expected results: 5.0
    loss_phase1 = loss_calculator(phase_context=context_phase1, network_outputs=network_outputs)
    assert torch.allclose(input=loss_phase1, other=torch.tensor(data=5.0)), \
        "Phase 1 accumulated loss does not match the configured weights."


def test_sir_vs_seirm_monotonicity_routing(base_config: EINNConfig):
    """
    Advanced architectural test: Ensures that SIR and SEIRM models apply
    monotonicity penalties to entirely different compartments based on their physics.
    """
    loss_sir = EINNLoss(config=base_config, ode_model=None, model_type="SIR")
    loss_seirm = EINNLoss(config=base_config, ode_model=None, model_type="SEIRM")

    # A derivative tensor where the value at index 2 is strongly negative (-5.0).
    # In the SIR model, this is R (Recovered), so it must be penalized.
    # In the SEIRM model, this is I (Infected), which can fluctuate, so it is NOT penalized (only indices 3 and 4 are).
    mock_dS_dt = torch.tensor(data=[[[0.0, 0.0, -5.0, 0.0, 0.0]]], dtype=torch.float32)

    sir_penalty = loss_sir.calc_monotonicity_loss(
        dS_dt=mock_dS_dt, inc_indices=loss_sir.mono_inc_indices, dec_indices=loss_sir.mono_dec_indices
    )
    seirm_penalty = loss_seirm.calc_monotonicity_loss(
        dS_dt=mock_dS_dt, inc_indices=loss_seirm.mono_inc_indices, dec_indices=loss_seirm.mono_dec_indices
    )

    # SIR penalty = (-(-5.0) * relu(5.0))^2 = 25.0 * 25.0 = 625.0
    assert sir_penalty.item() > 0.0, "SIR failed to penalize a decreasing Recovered compartment."
    assert seirm_penalty.item() == 0.0, "SEIRM incorrectly penalized the Infected compartment."


def test_parameter_smoothness_edge_case(loss_calculator_seirm: EINNLoss):
    """
    Edge case test: Ensures that parameter smoothness loss safely returns 0.0
    without crashing when the sequence length is 1.
    """
    # [Batch=1, Seq_len=1, d_p=4]
    params_seq_1 = torch.tensor(data=[[[0.5, 0.2, 0.1, 0.05]]], dtype=torch.float32)

    # It shouldn't give IndexError
    smooth_loss = loss_calculator_seirm.calc_parameter_smoothness_loss(params=params_seq_1)

    expected_loss = torch.tensor(data=0.0, dtype=torch.float32)
    assert torch.allclose(input=smooth_loss, other=expected_loss), \
        "Smoothness loss did not handle sequence length 1 gracefully."
