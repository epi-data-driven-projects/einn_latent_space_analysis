import pytest
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.loss.einn_loss import EINNLoss
from einn.model.einn_builder import EINNBuilder
from einn.model.interface.phase_context import PhaseContext
from einn.training.einn_forward_engine import EINNForwardEngine
from einn.training.phase_optimizers.phase_1_optimizer import Phase1Optimizer
from einn.training.phase_optimizers.phase_2_optimizer import Phase2Optimizer
from einn.training.phase_optimizers.phase_3_optimizer import Phase3Optimizer
from einn.training.phase_optimizers.phase_4_optimizer import Phase4Optimizer


@pytest.fixture
def integrated_components() -> dict:
    """
    Fixture providing models, engine, loss calculator, configuration, context,
    and scaling statistics for optimizer tests.

    :return dict: Instantiated components required by the tests.
    """
    model_config = EINNModelConfig(
        d_x=5,
        d_e=10,
        d_s=5,
        d_p=4,
        feature_n_layers=1,
    )
    train_config = EINNTrainConfig(device="cpu")

    scaling_stats = DataScalingStats(
        aux_mean=torch.zeros(5),
        aux_std=torch.ones(5),
        t_min=torch.tensor(0.0),
        t_max=torch.tensor(1.0),
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
    )

    models = EINNBuilder.build_einn(
        model_config=model_config,
        train_config=train_config,
        param_calibration={
            "beta": 0.4,
            "alpha": 0.2,
            "gamma": 0.1,
            "mu": 0.05,
        },
        model_type="SEIRM",
        seed=42,
        target_idx=2,
    )

    engine = EINNForwardEngine(
        train_config=train_config,
        scaling_stats=scaling_stats,
    )
    loss_calc = EINNLoss(
        config=train_config,
        ode_model=models.ode_model,
        scaling_stats=scaling_stats,
    )

    batch_size = 2
    past_steps = 7
    x_past = torch.rand(size=(batch_size, past_steps, 5))
    y_past = torch.rand(size=(batch_size, past_steps, 1))
    aux_targets = torch.rand(size=(batch_size, past_steps, 5))

    # The time tensor must have the same sequence length as x_past.
    t_past = (
        torch.linspace(0.0, 1.0, past_steps)
        .view(1, past_steps, 1)
        .expand(batch_size, -1, -1)
        .clone()
    )

    context = PhaseContext(
        phase_num=1,
        epoch=1,
        x=x_past,
        y=y_past,
        t=t_past,
        aux_targets=aux_targets,
        models=models,
    )

    return {
        "models": models,
        "config": train_config,
        "engine": engine,
        "loss_calc": loss_calc,
        "context": context,
    }


def _check_requires_grad(module: torch.nn.Module) -> bool:
    """
    Check whether a module has at least one trainable parameter.

    :param torch.nn.Module module: Module to inspect.
    :return bool: True if at least one parameter requires gradients.
    """
    return any(parameter.requires_grad for parameter in module.parameters())


def test_phase_one_optimizer_freezing_rules(
    integrated_components: dict,
) -> None:
    """
    Check that Phase 1 trains the Time and Output modules and freezes
    the Feature module and ODE parameters.
    """
    models = integrated_components["models"]

    optimizer = Phase1Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"],
    )
    optimizer.prepare_network_states()

    assert _check_requires_grad(models.time_module) is True
    assert _check_requires_grad(models.output_module) is True
    assert _check_requires_grad(models.feature_module) is False
    assert _check_requires_grad(models.ode_model) is False


def test_phase_two_optimizer_freezing_rules(
    integrated_components: dict,
) -> None:
    """
    Check that Phase 2 trains the Time, Output, and ODE components while
    freezing the Feature module.
    """
    models = integrated_components["models"]

    optimizer = Phase2Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"],
    )
    optimizer.prepare_network_states()

    assert _check_requires_grad(models.time_module) is True
    assert _check_requires_grad(models.output_module) is True
    assert _check_requires_grad(models.ode_model) is True
    assert _check_requires_grad(models.feature_module) is False


def test_phase_three_optimizer_freezing_rules(
    integrated_components: dict,
) -> None:
    """
    Check that Phase 3 trains the Time, Feature, and Output modules while
    freezing the ODE parameters.
    """
    models = integrated_components["models"]

    optimizer = Phase3Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"],
    )
    optimizer.prepare_network_states()

    assert _check_requires_grad(models.time_module) is True
    assert _check_requires_grad(models.feature_module) is True
    assert _check_requires_grad(models.output_module) is True
    assert _check_requires_grad(models.ode_model) is False


def test_phase_four_optimizer_freezing_rules(
    integrated_components: dict,
) -> None:
    """
    Check that Phase 4 trains the Output module and ODE parameters while
    freezing the Time and Feature modules.
    """
    models = integrated_components["models"]

    optimizer = Phase4Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"],
    )
    optimizer.prepare_network_states()

    assert _check_requires_grad(models.time_module) is False
    assert _check_requires_grad(models.feature_module) is False
    assert _check_requires_grad(models.output_module) is True
    assert _check_requires_grad(models.ode_model) is True


def test_optimizer_training_mode_toggle(
    integrated_components: dict,
) -> None:
    """
    Check that the optimizer switches active modules to training mode and
    frozen modules to evaluation mode.
    """
    models = integrated_components["models"]

    optimizer = Phase1Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"],
    )

    for model in [
        models.time_module,
        models.feature_module,
        models.output_module,
    ]:
        model.train()

    optimizer.prepare_network_states()

    assert models.time_module.training is True
    assert models.feature_module.training is False


def test_optimizer_parameter_group_fidelity(
    integrated_components: dict,
) -> None:
    """
    Check that the Phase 4 optimizer tracks only the ODE and Output parameters.
    """
    models = integrated_components["models"]

    optimizer = Phase4Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"],
    )

    expected_param_count = (
        len(list(models.ode_model.parameters()))
        + len(list(models.output_module.parameters()))
    )
    tracked_params = optimizer.optimizer.param_groups[0]["params"]

    assert len(tracked_params) == expected_param_count, (
        f"Adam tracks {len(tracked_params)} parameters, "
        f"but {expected_param_count} were expected."
    )


def test_optimizer_edge_case_uninitialized_step(
    integrated_components: dict,
) -> None:
    """
    Check that step() raises an error if the optimizer is uninitialized.
    """
    optimizer = Phase1Optimizer(
        models=integrated_components["models"],
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"],
    )

    optimizer.optimizer = None

    with pytest.raises(
        expected_exception=RuntimeError,
        match="Optimizer was not initialized",
    ):
        optimizer.step(context=integrated_components["context"])


def test_optimizer_full_integration_step(
    integrated_components: dict,
) -> None:
    """
    Run a complete Phase 1 forward, loss, backward, and optimizer step.
    """
    optimizer = Phase1Optimizer(
        models=integrated_components["models"],
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"],
    )
    optimizer.prepare_network_states()

    initial_weights = (
        integrated_components["models"]
        .output_module.net[0]
        .weight.clone()
    )

    loss_value, loss_dict = optimizer.step(
        context=integrated_components["context"]
    )

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0

    updated_weights = (
        integrated_components["models"]
        .output_module.net[0]
        .weight
    )
    assert not torch.equal(initial_weights, updated_weights), (
        "Optimizer did not update the model weights."
    )


def test_sequential_phase_optimizer_integration() -> None:
    """
    Run one training step through each of the four phase optimizers.
    """
    model_config = EINNModelConfig(
        d_x=5,
        d_e=10,
        d_s=5,
        d_p=4,
        feature_n_layers=1,
    )
    train_config = EINNTrainConfig(
        device="cpu",
        learning_rate=0.01,
    )

    scaling_stats = DataScalingStats(
        aux_mean=torch.zeros(5),
        aux_std=torch.ones(5),
        t_min=torch.tensor(0.0),
        t_max=torch.tensor(1.0),
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
    )

    models = EINNBuilder.build_einn(
        model_config=model_config,
        train_config=train_config,
        param_calibration={
            "beta": 0.4,
            "alpha": 0.2,
            "gamma": 0.1,
            "mu": 0.05,
        },
        model_type="SEIRM",
        seed=42,
        target_idx=2,
    )

    engine = EINNForwardEngine(
        train_config=train_config,
        scaling_stats=scaling_stats,
    )
    loss_calculator = EINNLoss(
        config=train_config,
        ode_model=models.ode_model,
        scaling_stats=scaling_stats,
    )

    optimizers = [
        Phase1Optimizer(
            models=models,
            config=train_config,
            engine=engine,
            loss_calculator=loss_calculator,
        ),
        Phase2Optimizer(
            models=models,
            config=train_config,
            engine=engine,
            loss_calculator=loss_calculator,
        ),
        Phase3Optimizer(
            models=models,
            config=train_config,
            engine=engine,
            loss_calculator=loss_calculator,
        ),
        Phase4Optimizer(
            models=models,
            config=train_config,
            engine=engine,
            loss_calculator=loss_calculator,
        ),
    ]

    batch_size = 2
    seq_len = 5

    x = torch.rand(size=(batch_size, seq_len, 5))
    y = torch.rand(size=(batch_size, seq_len, 1))
    t = (
        torch.linspace(0.0, 1.0, seq_len)
        .view(1, seq_len, 1)
        .expand(batch_size, -1, -1)
        .clone()
    )
    aux_targets = torch.rand(size=(batch_size, seq_len, 5))

    recorded_losses = []

    for phase_idx, optimizer in enumerate(optimizers, start=1):
        context = PhaseContext(
            phase_num=phase_idx,
            epoch=1,
            x=x,
            y=y,
            t=t,
            aux_targets=aux_targets,
            models=models,
        )

        try:
            loss_value, loss_dict = optimizer.step(context=context)
        except Exception as exc:
            pytest.fail(
                f"Phase {phase_idx} failed during sequential execution: {exc}"
            )

        assert not torch.isnan(torch.tensor(loss_value)), (
            f"Phase {phase_idx} resulted in a NaN loss."
        )
        recorded_losses.append(loss_value)

    assert len(recorded_losses) == 4

    assert next(models.output_module.parameters()).requires_grad is True
    assert next(models.time_module.parameters()).requires_grad is False
