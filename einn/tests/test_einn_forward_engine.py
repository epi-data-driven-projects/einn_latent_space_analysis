import pytest
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.model.einn_builder import EINNBuilder
from einn.model.interface.einn_models import EINNModels
from einn.model.interface.phase_context import PhaseContext
from einn.training.einn_forward_engine import EINNForwardEngine


@pytest.fixture
def scaling_stats() -> DataScalingStats:
    """
    Fixture providing neutral scaling parameters for tests.

    :return DataScalingStats: Initialized dummy scaling statistics.
    """
    return DataScalingStats(
        aux_mean=torch.zeros(5),
        aux_std=torch.ones(5),
        t_min=torch.tensor(0.0),
        t_max=torch.tensor(1.0),
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
    )


@pytest.fixture
def engine(scaling_stats: DataScalingStats) -> EINNForwardEngine:
    """
    Fixture providing an EINNForwardEngine configured for CPU execution.

    :param DataScalingStats scaling_stats: Scaling statistics injected into the engine.
    :return EINNForwardEngine: Initialized forward engine.
    """
    config = EINNTrainConfig(device="cpu")
    return EINNForwardEngine(
        train_config=config,
        scaling_stats=scaling_stats,
    )


@pytest.fixture
def initialized_models() -> EINNModels:
    """
    Fixture providing initialized neural networks and a SEIRM ODE model.

    :return EINNModels: Instantiated model components.
    """
    model_config = EINNModelConfig(
        d_x=5,
        d_e=10,
        d_s=5,
        d_p=4,
        feature_n_layers=1,
    )
    train_config = EINNTrainConfig(device="cpu")
    calib = {"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05}

    return EINNBuilder.build_einn(
        model_config=model_config,
        train_config=train_config,
        param_calibration=calib,
        model_type="SEIRM",
        target_idx=1,
        seed=42,
    )


def _make_past_time_grid(batch_size: int, past_steps: int) -> torch.Tensor:
    """
    Create a strictly increasing time grid for the observed sequence.

    :param int batch_size: Number of samples in the batch.
    :param int past_steps: Number of observed time points.
    :return torch.Tensor: Time tensor with shape [batch_size, past_steps, 1].
    """
    time_grid = torch.linspace(0.0, 1.0, past_steps).view(1, past_steps, 1)
    return time_grid.expand(batch_size, -1, -1).clone()


def test_empirical_derivative(engine: EINNForwardEngine) -> None:
    """
    Verify the empirical derivative calculation.

    If y = 3 * t^2, then dy/dt = 6 * t.

    :param EINNForwardEngine engine: Fixture providing the execution engine.
    """
    t = torch.tensor([[[2.0], [3.0]]], requires_grad=True)
    y = 3.0 * (t**2)

    expected_dy_dt = torch.tensor([[[12.0], [18.0]]])
    calculated_dy_dt = engine._compute_empirical_derivatives(
        outputs=y,
        time_tensor=t,
    )

    assert torch.allclose(calculated_dy_dt, expected_dy_dt), (
        "Empirical derivative failed basic mathematical validation."
    )


def test_feature_gradient_trick(engine: EINNForwardEngine) -> None:
    """
    Verify the chain rule used by the Feature module gradient trick.

    Let e = 4, ds/de = 6e, and de/dt = 2. Then ds/dt = 48.

    :param EINNForwardEngine engine: Fixture providing the execution engine.
    """
    de_dt = torch.tensor([[[2.0]]])
    e_t_f_grad = torch.tensor([[[4.0]]], requires_grad=True)
    s_t_f = 3.0 * (e_t_f_grad**2)

    calculated_ds_dt = engine._compute_feature_gradient_trick(
        s_t_f=s_t_f,
        e_t_f_grad=e_t_f_grad,
        de_dt=de_dt,
    )

    expected_ds_dt = torch.tensor([[[48.0]]])
    assert torch.allclose(calculated_ds_dt, expected_ds_dt), (
        "Feature gradient trick failed chain-rule mathematical validation."
    )


def test_forward_slicing_phase_2(
    engine: EINNForwardEngine,
    initialized_models: EINNModels,
) -> None:
    """
    Verify that Phase 2 returns separate past and future derivative tensors.

    The context supplies only observed time points. The engine extends the time
    grid using the configured number of future steps.

    :param EINNForwardEngine engine: Fixture providing the execution engine.
    :param EINNModels initialized_models: Fixture providing the built models.
    """
    batch_size = 2
    past_steps = 10
    future_steps = 5

    engine.config.future_steps = future_steps

    x_past = torch.rand(size=(batch_size, past_steps, 5))
    t_past = _make_past_time_grid(batch_size, past_steps)

    context = PhaseContext(
        phase_num=2,
        epoch=1,
        x=x_past,
        y=torch.rand(size=(batch_size, past_steps, 1)),
        t=t_past,
        aux_targets=torch.rand(size=(batch_size, past_steps, 5)),
        models=initialized_models,
    )

    out = engine.forward(context=context)

    assert out.s_t.shape == (batch_size, past_steps, 5), (
        "Past state tensor sliced incorrectly."
    )
    assert out.ds_dt_T_nn.shape == (batch_size, past_steps, 5), (
        "Past derivative tensor sliced incorrectly."
    )
    assert out.ds_dt_future_T_nn.shape == (batch_size, future_steps, 5), (
        "Future derivative tensor sliced incorrectly."
    )
    assert out.s_t_F is None, "Feature states should not exist in Phase 2."


def test_forward_full_pipeline_phase_4(
    engine: EINNForwardEngine,
    initialized_models: EINNModels,
) -> None:
    """
    Verify the Phase 4 forward pass, including Feature gradients and slicing.

    :param EINNForwardEngine engine: Fixture providing the execution engine.
    :param EINNModels initialized_models: Fixture providing the built models.
    """
    batch_size = 2
    past_steps = 7
    future_steps = 3

    engine.config.future_steps = future_steps

    x_past = torch.rand(size=(batch_size, past_steps, 5))
    t_past = _make_past_time_grid(batch_size, past_steps)

    context = PhaseContext(
        phase_num=4,
        epoch=10,
        x=x_past,
        y=torch.rand(size=(batch_size, past_steps, 1)),
        t=t_past,
        aux_targets=torch.rand(size=(batch_size, past_steps, 5)),
        models=initialized_models,
    )

    out = engine.forward(context=context)

    assert out.ds_dt_F_nn is not None, "Feature gradient trick failed."
    assert out.ds_dt_F_nn.shape == (batch_size, past_steps, 5), (
        "Past Feature derivative shape mismatch."
    )
    assert out.ds_dt_future_F_nn is not None, (
        "Future Feature gradient trick failed."
    )
    assert out.ds_dt_future_F_nn.shape == (batch_size, future_steps, 5), (
        "Future Feature derivative shape mismatch."
    )

    ds_dt_sum = out.ds_dt_T_ode.sum(dim=-1)
    expected_zeros = torch.zeros_like(ds_dt_sum)
    assert torch.allclose(ds_dt_sum, expected_zeros, atol=1e-5), (
        "Closed ODE system mass conservation violated in engine forward pass."
    )