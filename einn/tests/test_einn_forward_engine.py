import pytest
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.model.einn_builder import EINNBuilder
from einn.model.interface.einn_models import EINNModels
from einn.model.interface.phase_context import PhaseContext
from einn.training.einn_forward_engine import EINNForwardEngine


@pytest.fixture
def engine() -> EINNForwardEngine:
    """
    Fixture providing the EINNForwardEngine initialized with a base CPU configuration.

    :return EINNForwardEngine: An initialized instance of the forward engine.
    """
    config = EINNTrainConfig(device='cpu')
    return EINNForwardEngine(train_config=config)


@pytest.fixture
def initialized_models() -> EINNModels:
    """
    Fixture providing fully initialized neural networks and ODE model (SEIRM).

    :return EINNModels: Dataclass containing the instantiated model components.
    """
    model_config = EINNModelConfig(d_x=5, d_e=10, d_s=5, d_p=4, feature_n_layers=1)
    train_config = EINNTrainConfig(device='cpu')
    calib = {"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05}

    return EINNBuilder.build_einn(
        model_config=model_config,
        train_config=train_config,
        param_calibration=calib,
        model_type="SEIRM",
        seed=42
    )


def test_empirical_derivative(engine: EINNForwardEngine):
    """
    Verifies the _compute_empirical_derivatives method.
    If y = 3 * t^2, then dy/dt = 6 * t.

    :param EINNForwardEngine engine: Fixture providing the execution engine.
    """
    # [Batch=1, Seq=2, Features=1]
    t = torch.tensor([[[2.0], [3.0]]], requires_grad=True)

    # Calculate y = 3 * t^2
    y = 3.0 * (t ** 2)

    # Expected derivative is 6 * t -> 12.0 and 18.0
    expected_dy_dt = torch.tensor([[[12.0], [18.0]]])

    calculated_dy_dt = engine._compute_empirical_derivatives(outputs=y, time_tensor=t)

    assert torch.allclose(input=calculated_dy_dt, other=expected_dy_dt), \
        "Empirical derivative failed basic mathematical validation."


def test_feature_gradient_trick(engine: EINNForwardEngine):
    """
    Verifies the chain-rule in _compute_feature_gradient_trick.
    Let embedding e = 2 * t. Then de/dt = 2.
    Let state s = 3 * e^2. Then ds/de = 6 * e.
    By chain rule, ds/dt = ds/de * de/dt = (6 * e) * 2 = 12 * e.
    If we evaluate at e = 4.0, ds/dt should be 48.0.

    :param EINNForwardEngine engine: Fixture providing the execution engine.
    """
    # Mock de/dt = 2.0
    de_dt = torch.tensor([[[2.0]]])

    # Mock e = 4.0 (requires grad for ds/de computation)
    e_t_f_grad = torch.tensor([[[4.0]]], requires_grad=True)

    # Mock s = 3 * e^2 (Value will be 3 * 16 = 48)
    s_t_f = 3.0 * (e_t_f_grad ** 2)

    # Apply the gradient trick
    calculated_ds_dt = engine._compute_feature_gradient_trick(
        s_t_f=s_t_f, e_t_f_grad=e_t_f_grad, de_dt=de_dt
    )

    # Expected: 12 * e = 12 * 4.0 = 48.0
    expected_ds_dt = torch.tensor([[[48.0]]])

    assert torch.allclose(input=calculated_ds_dt, other=expected_ds_dt), \
        "Feature gradient trick failed chain-rule mathematical validation."


def test_forward_slicing_phase_2(engine: EINNForwardEngine, initialized_models):
    """
    Verifies if the Engine correctly slices a unified time tensor into past and future chunks when processing Phase 2.

    :param EINNForwardEngine engine: Fixture providing the execution engine.
    :param EINNModels initialized_models: Fixture providing the built models.
    """
    past_steps = 10
    future_steps = 5
    total_steps = past_steps + future_steps

    # t represents the full sequence (past + future)
    t_full = torch.rand(size=(2, total_steps, 1))

    # X only represents past observations
    x_past = torch.rand(size=(2, past_steps, 5))

    context = PhaseContext(
        phase_num=2,
        epoch=1,
        x=x_past,
        y=torch.rand(size=(2, past_steps, 1)),
        t=t_full,
        aux_targets=torch.rand(size=(2, past_steps, 5)),
        models=initialized_models
    )

    out = engine.forward(context=context)

    # Check if past tensors have seq_len == past_steps
    assert out.s_t.shape == (2, past_steps, 5), "Past state tensor sliced incorrectly."
    assert out.ds_dt_T_nn.shape == (2, past_steps, 5), "Past derivative tensor sliced incorrectly."

    # Check if future tensors have seq_len == future_steps
    assert out.ds_dt_future_T_nn.shape == (2, future_steps, 5), "Future derivative tensor sliced incorrectly."

    # Feature module should be None in Phase 2
    assert out.s_t_F is None, "Feature states should not exist in Phase 2."


def test_forward_full_pipeline_phase_4(engine: EINNForwardEngine, initialized_models):
    """
    Executes the most complex phase (4), ensuring all modules, the gradient trick, and slicing work well

    :param EINNForwardEngine engine: Fixture providing the execution engine.
    :param EINNModels initialized_models: Fixture providing the built models.
    """
    past_steps = 7
    future_steps = 3
    total_steps = past_steps + future_steps

    t_full = torch.rand(size=(2, total_steps, 1))
    x_past = torch.rand(size=(2, past_steps, 5))

    context = PhaseContext(
        phase_num=4,
        epoch=10,
        x=x_past,
        y=torch.rand(size=(2, past_steps, 1)),
        t=t_full,
        aux_targets=torch.rand(size=(2, past_steps, 5)),
        models=initialized_models
    )

    out = engine.forward(context=context)

    # Verify Feature Module gradients were computed for both past and future
    assert out.ds_dt_F_nn is not None, "Feature gradient trick failed."
    assert out.ds_dt_F_nn.shape == (2, past_steps, 5), "Past Feature derivative shape mismatch."

    assert out.ds_dt_future_F_nn is not None, "Future Feature gradient trick failed."
    assert out.ds_dt_future_F_nn.shape == (2, future_steps, 5), "Future Feature derivative shape mismatch."

    # Physics check for the ODE derivative
    ds_dt_sum = out.ds_dt_T_ode.sum(dim=-1)
    expected_zeros = torch.zeros_like(ds_dt_sum)
    assert torch.allclose(input=ds_dt_sum, other=expected_zeros, atol=1e-5), \
        "Closed ODE system mass conservation violated in engine forward pass."
