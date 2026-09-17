import pytest
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
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
    Fixture providing all necessary components (Models, Engine, Loss, Configs, Context)
    to perform integration tests on the phase optimizers.

    :return dict: Dictionary holding all instantiated components required for testing.
    """
    model_config = EINNModelConfig(d_x=5, d_e=10, d_s=5, d_p=4, feature_n_layers=1)
    train_config = EINNTrainConfig(device='cpu')

    # Build the neural networks and ODE models
    models = EINNBuilder.build_einn(
        model_config=model_config,
        train_config=train_config,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
        model_type="SEIRM",
        seed=42
    )

    engine = EINNForwardEngine(train_config=train_config)
    loss_calc = EINNLoss(config=train_config, ode_model=models.ode_model, model_type="SEIRM")

    # Mock Data for the PhaseContext
    t_full = torch.rand(size=(2, 10, 1))
    x_past = torch.rand(size=(2, 7, 5))
    y_past = torch.rand(size=(2, 7, 1))

    context = PhaseContext(
        phase_num=1,
        epoch=1,
        X=x_past,
        y=y_past,
        t=t_full,
        aux_targets=torch.rand(size=(2, 7, 5)),
        models=models
    )

    return {
        "models": models,
        "config": train_config,
        "engine": engine,
        "loss_calc": loss_calc,
        "context": context
    }


def _check_requires_grad(module: torch.nn.Module) -> bool:
    """
    Helper method to verify if a PyTorch module has at least one active, trainable parameter.

    :param torch.nn.Module module: The neural network module to inspect.
    :return bool: True if the module is currently trainable (requires_grad=True), False otherwise.
    """
    return any(p.requires_grad for p in module.parameters())


def test_phase_one_optimizer_freezing_rules(integrated_components: dict):
    """
    Tests if Phase1Optimizer correctly initializes the Adam optimizer for the Time and Output modules,
    and ensures the Feature Module and ODE parameters are completely frozen.

    :param dict integrated_components: The fixture containing the model and optimizer configurations.
    """
    models = integrated_components["models"]

    opt1 = Phase1Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"]
    )

    # Apply the freezing rules
    opt1.prepare_network_states()

    assert _check_requires_grad(models.time_module) is True, "Time module must be active in Phase 1."
    assert _check_requires_grad(models.output_module) is True, "Output module must be active in Phase 1."
    assert _check_requires_grad(models.feature_module) is False, "Feature module MUST be frozen in Phase 1."
    assert _check_requires_grad(models.ode_model) is False, "ODE params MUST be frozen in Phase 1."


def test_phase_two_optimizer_freezing_rules(integrated_components: dict):
    """
    Tests if Phase2Optimizer correctly unfreezes ODE parameters alongside the Time and Output modules.

    :param dict integrated_components: The fixture containing the model and optimizer configurations.
    """
    models = integrated_components["models"]

    opt2 = Phase2Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"]
    )

    opt2.prepare_network_states()

    assert _check_requires_grad(models.time_module) is True, "Time module must be active in Phase 2."
    assert _check_requires_grad(models.output_module) is True, "Output module must be active in Phase 2."
    assert _check_requires_grad(models.ode_model) is True, "ODE params must be active in Phase 2."
    assert _check_requires_grad(models.feature_module) is False, "Feature module MUST be frozen in Phase 2."


def test_phase_three_optimizer_freezing_rules(integrated_components: dict):
    """
    Tests if Phase3Optimizer appropriately configures the networks for Knowledge Distillation.
    The ODE params are frozen, but Time, Feature, and Output modules remain active.

    :param dict integrated_components: The fixture containing the model and optimizer configurations.
    """
    models = integrated_components["models"]

    opt3 = Phase3Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"]
    )

    opt3.prepare_network_states()

    assert _check_requires_grad(models.time_module) is True, "Time module must be active in Phase 3."
    assert _check_requires_grad(models.feature_module) is True, "Feature module must be active in Phase 3."
    assert _check_requires_grad(models.output_module) is True, "Output module must be active in Phase 3."
    assert _check_requires_grad(models.ode_model) is False, "ODE params MUST be frozen in Phase 3."


def test_phase_four_optimizer_freezing_rules(integrated_components: dict):
    """
    Tests if Phase4Optimizer strictly isolates the gradient flow to the Output Module
    and ODE params, completely freezing the latent embeddings (Time and Feature modules).

    :param dict integrated_components: The fixture containing the model and optimizer configurations.
    """
    models = integrated_components["models"]

    opt4 = Phase4Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"]
    )

    opt4.prepare_network_states()

    assert _check_requires_grad(models.time_module) is False, "Time module MUST be frozen in Phase 4."
    assert _check_requires_grad(models.feature_module) is False, "Feature module MUST be frozen in Phase 4."
    assert _check_requires_grad(models.output_module) is True, "Output module must be active in Phase 4."
    assert _check_requires_grad(models.ode_model) is True, "ODE params must be active in Phase 4."


def test_optimizer_training_mode_toggle(integrated_components: dict):
    """
    Verifies that the base optimizer not only toggles 'requires_grad',
    but correctly sets '.train()' and '.eval()' modes for elements like Dropout and BatchNorm.

    :param dict integrated_components: The fixture containing the model and optimizer configurations.
    """
    models = integrated_components["models"]

    opt1 = Phase1Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"]
    )

    # Forcefully set everything to train mode beforehand
    for model in [models.time_module, models.feature_module, models.output_module]:
        model.train()

    opt1.prepare_network_states()

    # Feature module should be switched to eval() mode during Phase 1
    assert models.time_module.training is True, "Active modules must be in training mode."
    assert models.feature_module.training is False, "Frozen modules MUST be switched to eval mode."


def test_optimizer_parameter_group_fidelity(integrated_components: dict):
    """
    Verifies that the internal Adam optimizer instances precisely track
    only the parameters designated by the specific phase via itertools.chain.

    :param dict integrated_components: The fixture containing the model and optimizer configurations.
    """
    models = integrated_components["models"]

    # In Phase 4, only the ODE model and the Output module should be passed to the optimizer.
    opt4 = Phase4Optimizer(
        models=models,
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"]
    )

    # Calculate the exact number of parameter tensors expected
    expected_param_count = len(list(models.ode_model.parameters())) + len(list(models.output_module.parameters()))

    # Extract the actual number of parameters tracked by Adam
    tracked_params = opt4.optimizer.param_groups[0]['params']

    assert len(tracked_params) == expected_param_count, \
        f"Adam optimizer tracks {len(tracked_params)} parameters, but exactly {expected_param_count} were expected."


def test_optimizer_edge_case_uninitialized_step(integrated_components: dict):
    """
    Ensures a RuntimeError is safely raised if `step()` is called
    before the networks are prepared and the optimizer is configured.

    :param dict integrated_components: The fixture containing the model and optimizer configurations.
    """
    opt1 = Phase1Optimizer(
        models=integrated_components["models"],
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"]
    )

    # Forcefully detach the optimizer to simulate an uninitialized state
    opt1.optimizer = None

    with pytest.raises(expected_exception=RuntimeError, match="Optimizer was not initialized"):
        opt1.step(context=integrated_components["context"])


def test_optimizer_full_integration_step(integrated_components: dict):
    """
    Runs a complete training iteration (Forward -> Loss -> Backward -> Step)
    using the Phase1Optimizer to ensure gradients flow correctly and weights are updated.

    :param dict integrated_components: The fixture containing the model and optimizer configurations.
    """
    opt1 = Phase1Optimizer(
        models=integrated_components["models"],
        config=integrated_components["config"],
        engine=integrated_components["engine"],
        loss_calculator=integrated_components["loss_calc"]
    )

    opt1.prepare_network_states()

    # Clone the initial weights of the Output Module for comparison
    initial_weights = integrated_components["models"].output_module.net[0].weight.clone()

    # Execute a full optimization step
    loss_value = opt1.step(context=integrated_components["context"])

    # Validations
    assert isinstance(loss_value, float), "The step method must return a float loss value."
    assert loss_value >= 0.0, "The calculated loss value must be non-negative."

    updated_weights = integrated_components["models"].output_module.net[0].weight

    # Prove that optimizer.step() successfully mutated the network's weights
    assert not torch.equal(input=initial_weights, other=updated_weights), \
        "Optimizer failed to update the model weights during the training step."
