import pytest
import torch

from einn.config.einn_train_config import EINNTrainConfig
from einn.loss.einn_loss import EINNLoss
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext
from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel


@pytest.fixture
def base_train_config() -> EINNTrainConfig:
    """
    Fixture providing a base training configuration on CPU.

    :return EINNTrainConfig: Training configuration instance.
    """
    return EINNTrainConfig(device='cpu')


@pytest.fixture
def loss_calculator_seirm(base_train_config: EINNTrainConfig) -> EINNLoss:
    """
    Fixture providing an initialized EINNLoss configured for the SEIRM model.

    :param EINNTrainConfig base_train_config: The base training configuration fixture.
    :return EINNLoss: Initialized loss calculator using SEIRM monotonicity rules.
    """
    ode_model = SEIRMModel(population_n=1.0)
    return EINNLoss(config=base_train_config, ode_model=ode_model)


def test_monotonicity_loss_penalties(loss_calculator_seirm: EINNLoss):
    """
    Tests if the asymmetric squared ReLU correctly penalizes violating derivatives.
    For SEIRM: S (idx 0) must decrease, R (idx 3) and M (idx 4) must increase.

    :param EINNLoss loss_calculator_seirm: The SEIRM loss calculator fixture.
    """
    # Test case: S is improperly increasing (+2.0), R is improperly decreasing (-3.0), M is valid (0.0)
    ds_dt = torch.tensor(data=[[[2.0, 0.0, 0.0, -3.0, 0.0]]], dtype=torch.float32)

    mono_loss = loss_calculator_seirm.calc_monotonicity_loss(
        ds_dt=ds_dt,
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

    :param EINNLoss loss_calculator_seirm: The SEIRM loss calculator fixture.
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
    Tests if the cascading forward pass correctly aggregates active losses
    when executing the final phase (Phase 4), strictly using the NetworkOutputs dataclass.

    :param EINNLoss loss_calculator_seirm: The SEIRM loss calculator fixture.
    """
    mock_tensor = torch.zeros(size=(1, 5, 5))
    mock_params = torch.zeros(size=(1, 5, 4))

    network_outputs = NetworkOutputs(
        s_t=mock_tensor,
        s_t_F=mock_tensor,
        e_t=mock_tensor,
        e_t_F=mock_tensor,
        ds_dt_T_nn=mock_tensor,
        ds_dt_T_ode=mock_tensor,
        ds_dt_future_T_nn=mock_tensor,
        ds_dt_future_T_ode=mock_tensor,
        ds_dt_F_nn=mock_tensor,
        ds_dt_F_ode=mock_tensor,
        ds_dt_future_F_nn=mock_tensor,
        ds_dt_future_F_ode=mock_tensor,
        params=mock_params
    )

    context = PhaseContext(
        phase_num=4,
        epoch=1,
        x=mock_tensor,
        y=mock_tensor,
        t=torch.zeros(size=(1, 5, 1)),
        aux_targets=mock_tensor,
        models=None
    )

    loss = loss_calculator_seirm(phase_context=context, network_outputs=network_outputs)
    assert isinstance(loss, torch.Tensor), "Forward pass must return a Tensor."
    assert not torch.isnan(input=loss), "Loss computation resulted in NaN."


def test_phase_routing_and_weight_integration(base_train_config: EINNTrainConfig):
    """
    Advanced combined test: Verifies that loss weights are correctly applied and
    that the cascading phase logic perfectly accumulates the expected losses.

    :param EINNTrainConfig base_train_config: The training configuration fixture.
    """
    base_train_config.loss_weights = {
        'data_T': 2.0, 'aux': 3.0, 'mono': 0.0, 'ode_T': 0.0, 'ode_future_T': 0.0,
        'param': 0.0, 'data_F': 0.0, 'kd_target': 0.0, 'kd_emb': 0.0, 'ode_F': 0.0, 'ode_future_F': 0.0
    }

    ode_model = SIRModel()
    loss_calculator = EINNLoss(config=base_train_config, ode_model=ode_model)

    # MSE should be 1.0
    mock_targets = torch.zeros(size=(1, 5, 1))
    mock_states = torch.ones(size=(1, 5, 3))

    network_outputs = NetworkOutputs(
        s_t=mock_states, s_t_F=mock_states, e_t=mock_states, e_t_F=mock_states,
        ds_dt_T_nn=mock_states, ds_dt_T_ode=mock_states, ds_dt_future_T_nn=mock_states,
        ds_dt_future_T_ode=mock_states, ds_dt_F_nn=mock_states,
        ds_dt_F_ode=mock_states, ds_dt_future_F_nn=mock_states, ds_dt_future_F_ode=mock_states,
        params=torch.ones(size=(1, 5, 2))
    )

    context_phase1 = PhaseContext(
        phase_num=1, epoch=1, x=mock_targets, y=mock_targets,
        t=torch.zeros(size=(1, 5, 1)), aux_targets=mock_targets, models=None
    )

    # In Phase 1: data_T (2.0) + aux (3.0) -> Expected results: 5.0
    loss_phase1 = loss_calculator(phase_context=context_phase1, network_outputs=network_outputs)
    assert torch.allclose(input=loss_phase1, other=torch.tensor(data=5.0)), \
        "Phase 1 accumulated loss does not match the configured weights."


def test_sir_vs_seirm_monotonicity_routing(base_train_config: EINNTrainConfig):
    """
    Advanced architectural test: Ensures that SIR and SEIRM models apply
    monotonicity penalties to entirely different compartments based on their internal physics.

    :param EINNTrainConfig base_train_config: The training configuration fixture.
    """
    loss_sir = EINNLoss(config=base_train_config, ode_model=SIRModel())
    loss_seirm = EINNLoss(config=base_train_config, ode_model=SEIRMModel(population_n=1.0))

    # A derivative tensor where the value at index 2 is strongly negative (-5.0).
    # In the SIR model, this is R (Recovered), so it must be penalized.
    # In the SEIRM model, this is I (Infected), which can fluctuate, so it is NOT penalized.
    mock_ds_dt = torch.tensor(data=[[[0.0, 0.0, -5.0, 0.0, 0.0]]], dtype=torch.float32)

    sir_penalty = loss_sir.calc_monotonicity_loss(
        ds_dt=mock_ds_dt, inc_indices=loss_sir.mono_inc_indices, dec_indices=loss_sir.mono_dec_indices
    )
    seirm_penalty = loss_seirm.calc_monotonicity_loss(
        ds_dt=mock_ds_dt, inc_indices=loss_seirm.mono_inc_indices, dec_indices=loss_seirm.mono_dec_indices
    )

    # SIR penalty = (-(-5.0) * relu(5.0))^2 = 25.0 * 25.0 = 625.0
    assert sir_penalty.item() > 0.0, "SIR failed to penalize a decreasing Recovered compartment."
    assert seirm_penalty.item() == 0.0, "SEIRM incorrectly penalized the Infected compartment."


def test_parameter_smoothness_edge_case(loss_calculator_seirm: EINNLoss):
    """
    Ensures that parameter smoothness loss safely returns 0.0
    without crashing when the sequence length is exactly 1.

    :param EINNLoss loss_calculator_seirm: The SEIRM loss calculator fixture.
    """
    # [Batch=1, Seq_len=1, d_p=4]
    params_seq_1 = torch.tensor(data=[[[0.5, 0.2, 0.1, 0.05]]], dtype=torch.float32)

    smooth_loss = loss_calculator_seirm.calc_parameter_smoothness_loss(params=params_seq_1)

    expected_loss = torch.tensor(data=0.0, dtype=torch.float32)
    assert torch.allclose(input=smooth_loss, other=expected_loss), \
        "Smoothness loss did not handle sequence length 1 gracefully."


def test_knowledge_distillation_gradient_isolation(loss_calculator_seirm: EINNLoss):
    """
    Proves mathematically that the Knowledge Distillation (KD) loss component does not leak gradients into the
    Time module, even when the Time Module is actively learning and receiving gradients from
    other active losses in Phase 3.

    :param EINNLoss loss_calculator_seirm: The SEIRM loss calculator fixture.
    """
    # Time Module outputs (requires_grad=True to simulate the active Time Module)
    teacher_s_t = torch.rand(size=(1, 5, 5), requires_grad=True)
    teacher_e_t = torch.rand(size=(1, 5, 20), requires_grad=True)

    # Feature Module outputs (requires_grad=True to simulate the active Feature Module)
    student_s_t_F = torch.rand(size=(1, 5, 5), requires_grad=True)
    student_e_t_F = torch.rand(size=(1, 5, 20), requires_grad=True)

    # Dummy tensors to satisfy the Phase 1 and 2 cascades without error
    dummy_states = torch.zeros(size=(1, 5, 5))
    dummy_params = torch.zeros(size=(1, 5, 4))

    def run_forward_backward(kd_weight: float):
        """
        Helper function to run a full forward-backward pass and return the cloned gradients.
        """
        # Reset gradients manually
        teacher_s_t.grad = None
        teacher_e_t.grad = None
        student_s_t_F.grad = None
        student_e_t_F.grad = None

        network_outputs = NetworkOutputs(
            s_t=teacher_s_t, e_t=teacher_e_t,
            s_t_F=student_s_t_F, e_t_F=student_e_t_F,
            ds_dt_T_nn=dummy_states, ds_dt_T_ode=dummy_states,
            ds_dt_future_T_nn=dummy_states, ds_dt_future_T_ode=dummy_states,
            ds_dt_F_nn=dummy_states, ds_dt_F_ode=dummy_states,
            ds_dt_future_F_nn=dummy_states, ds_dt_future_F_ode=dummy_states,
            params=dummy_params
        )

        context = PhaseContext(
            phase_num=3, epoch=1,
            x=torch.zeros(size=(1, 5, 5)),
            y=torch.zeros(size=(1, 5, 1)),
            t=torch.zeros(size=(1, 5, 1)),
            aux_targets=dummy_states,
            models=None
        )

        # Set active baseline losses so the Time Module natively receives gradients
        loss_calculator_seirm.weights['data_T'] = 1.0
        loss_calculator_seirm.weights['aux'] = 1.0

        # Toggle Knowledge Distillation dynamically
        loss_calculator_seirm.weights['kd_target'] = kd_weight
        loss_calculator_seirm.weights['kd_emb'] = kd_weight

        total_loss = loss_calculator_seirm(phase_context=context, network_outputs=network_outputs)
        total_loss.backward()

        return (
            teacher_s_t.grad.clone() if teacher_s_t.grad is not None else None,
            student_s_t_F.grad.clone() if student_s_t_F.grad is not None else None,
            student_e_t_F.grad.clone() if student_e_t_F.grad is not None else None
        )

    # Run WITHOUT Knowledge Distillation (Baseline gradients)
    t_s_grad_base, s_s_grad_base, s_e_grad_base = run_forward_backward(kd_weight=0.0)

    # Run WITH Knowledge Distillation
    t_s_grad_kd, s_s_grad_kd, s_e_grad_kd = run_forward_backward(kd_weight=1.0)

    # The Time Module MUST have received base gradients from data_T and aux losses
    assert t_s_grad_base is not None, "Time Module did not receive baseline gradients."

    # The Time Module's gradients MUST be mathematically identical whether KD is applied or not.
    assert torch.allclose(input=t_s_grad_base, other=t_s_grad_kd), \
        "Knowledge Distillation leaked into the Times Module's state gradients despite .detach()!"

    # The Feature Module MUST receive new, distinct gradients when KD is turned on.
    assert s_s_grad_kd is not None, "Feature Module states did not receive gradients from KD."
    assert s_e_grad_kd is not None, "Feature Module embeddings did not receive gradients from KD."


def test_all_loss_components_aggregation(loss_calculator_seirm: EINNLoss):
    """
    Calculates and aggregates all 11 loss components during a Phase 4 forward pass.
    Validates the mathematical correctness of every single static loss function (Data, Aux, ODE, Monotonicity,
    Param smoothness, KD) simultaneously.

    :param EINNLoss loss_calculator_seirm: The SEIRM loss calculator fixture.
    """
    # 1. Minden kulcs csupa kisbetű, és bekerült mindkét (t és f) ode_future!
    loss_calculator_seirm.weights = {
        'data_t': 1.0, 'aux': 1.0, 'mono': 1.0,
        'ode_t': 1.0, 'ode_future_t': 1.0, 'param': 1.0,
        'data_f': 1.0, 'kd_target': 1.0, 'kd_emb': 1.0,
        'ode_f': 1.0, 'ode_future_f': 1.0
    }

    # Tensors for Data, Aux and Knowledge Distillation
    # data_t: MSE(2.0, 1.0) = 1.0
    # aux: MSE(2.0, 1.0) = 1.0
    # data_f: MSE(3.0, 1.0) = 4.0
    # kd_target: MSE(3.0, 2.0) = 1.0
    # kd_emb: MSE(4.0, 5.0) = 1.0
    s_t = torch.full(size=(1, 3, 5), fill_value=2.0)
    y = torch.full(size=(1, 3, 5), fill_value=1.0)
    aux_targets = torch.full(size=(1, 3, 5), fill_value=1.0)

    s_t_f = torch.full(size=(1, 3, 5), fill_value=3.0)
    e_t = torch.full(size=(1, 3, 10), fill_value=5.0)
    e_t_f = torch.full(size=(1, 3, 10), fill_value=4.0)

    # Tensors for ODE "physics" (Past & Future)
    # ode_t: MSE(1.0, 2.0) = 1.0
    # ode_future_t: MSE(1.0, 0.0) = 1.0
    # ode_f: MSE(5.0, 3.0) = 4.0
    # ode_future_f: MSE(3.0, 3.0) = 0.0
    ds_dt_t_ode = torch.full(size=(1, 3, 5), fill_value=2.0)
    ds_dt_t_nn = torch.full(size=(1, 3, 5), fill_value=1.0)

    ds_dt_future_t_ode = torch.full(size=(1, 3, 5), fill_value=0.0)
    ds_dt_future_t_nn = torch.full(size=(1, 3, 5), fill_value=1.0)

    ds_dt_f_ode = torch.full(size=(1, 3, 5), fill_value=3.0)
    ds_dt_f_nn = torch.full(size=(1, 3, 5), fill_value=5.0)

    ds_dt_future_f_ode = torch.full(size=(1, 3, 5), fill_value=3.0)
    ds_dt_future_f_nn = torch.full(size=(1, 3, 5), fill_value=3.0)

    # Monotonicity Loss
    # SEIRM inc_indices [3, 4] MUST increase. 2.0 is > 0 -> Penalty: 0.0
    # SEIRM dec_indices [0] MUST decrease. 2.0 is NOT < 0.
    # Penalty calculation: (val * relu(val))^2 -> (2.0 * 2.0)^2 = 16.0
    # mono = 16.0

    # Parameter Smoothness Loss
    # Sequence of 3 timesteps: 1.0 -> 2.0 -> 3.0.
    # Squared differences are 1.0. Mean is 1.0.
    # param = 1.0
    params = torch.tensor(data=[[
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0]
    ]], dtype=torch.float32)

    network_outputs = NetworkOutputs(
        s_t=s_t, e_t=e_t, s_t_F=s_t_f, e_t_F=e_t_f,
        ds_dt_T_nn=ds_dt_t_nn, ds_dt_T_ode=ds_dt_t_ode,
        ds_dt_future_T_nn=ds_dt_future_t_nn, ds_dt_future_T_ode=ds_dt_future_t_ode,
        ds_dt_F_nn=ds_dt_f_nn, ds_dt_F_ode=ds_dt_f_ode,
        ds_dt_future_F_nn=ds_dt_future_f_nn, ds_dt_future_F_ode=ds_dt_future_f_ode,
        params=params
    )

    phase_context = PhaseContext(
        phase_num=4, epoch=1,
        x=torch.zeros(size=(1, 3, 5)),
        y=y,
        t=torch.zeros(size=(1, 3, 1)),
        aux_targets=aux_targets,
        models=None
    )

    total_loss = loss_calculator_seirm(phase_context=phase_context, network_outputs=network_outputs)

    # 15.0 (Standard Losses) + 16.0 (Monotonicity) = 31.0
    expected_total = torch.tensor(data=31.0, dtype=torch.float32)

    assert torch.allclose(input=total_loss, other=expected_total, atol=1e-5), \
        f"Aggregated loss was {total_loss.item()}, but expected {expected_total.item()}."
