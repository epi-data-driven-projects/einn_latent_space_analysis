import pytest
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.einn_dataset import EINNDataset
from einn.model.einn_builder import EINNBuilder
from einn.model.interface.training_metrics import TrainingMetrics
from einn.training.einn_trainer import EINNTrainer


@pytest.fixture
def einn_configs() -> dict:
    """
    Fixture providing the baseline model and training configurations.
    Separating this allows other tests to easily access configs without instantiating the full trainer.

    :return dict: Dictionary containing the model and training configurations.
    """
    model_config = EINNModelConfig(d_x=5, d_e=10, d_s=5, d_p=4, feature_n_layers=1)

    # Default repetitions - these can be dynamically overwritten in specific tests
    default_reps = {
        'phase_1': 1,
        'phase_2': 1,
        'phase_3': 1,
        'phase_4': 1
    }

    # Now injecting the 'reps' directly into the configuration as requested
    train_config = EINNTrainConfig(device='cpu', learning_rate=0.01, reps=default_reps)

    return {
        "model_config": model_config,
        "train_config": train_config
    }


@pytest.fixture
def integrated_trainer_setup(einn_configs: dict) -> dict:
    """
    Fixture providing all initialized components needed to test the EINNTrainer, including a mock dataset.
    Uses the separated einn_configs fixture.

    :param dict einn_configs: The fixture providing the configuration objects.
    :return dict: Dictionary containing the trainer, dataset, and configs.
    """
    model_config = einn_configs["model_config"]
    train_config = einn_configs["train_config"]

    models = EINNBuilder.build_einn(
        model_config=model_config,
        train_config=train_config,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
        model_type="SEIRM",
        seed=42
    )

    trainer = EINNTrainer(models=models, config=train_config)

    seq_len = 15
    dummy_dataset = EINNDataset(
        x=torch.rand(size=(seq_len, 5)),
        y=torch.zeros(size=(seq_len, 1)),
        t=torch.linspace(0, 1, seq_len).unsqueeze(1),
        aux_targets=torch.zeros(size=(seq_len, 5)),
        window_size=10,
        device='cpu'
    )

    return {
        "trainer": trainer,
        "dataset": dummy_dataset,
        "config": train_config
    }


def test_trainer_initialization(integrated_trainer_setup: dict):
    """
    Tests if the EINNTrainer correctly initializes all independent phase optimizers and preserves their
    unique Adam optimizer states.

    :param dict integrated_trainer_setup: The fixture providing the trainer.
    """
    trainer = integrated_trainer_setup["trainer"]

    assert trainer.opt_phase1 is not None, "Phase 1 optimizer was not initialized."
    assert trainer.opt_phase4 is not None, "Phase 4 optimizer was not initialized."

    assert id(trainer.opt_phase1.optimizer) != id(trainer.opt_phase2.optimizer), \
        "Phase optimizers must maintain independent Adam states (different memory addresses)."


def test_trainer_phase_routing_and_reps(integrated_trainer_setup: dict):
    """
    Verifies that the train() method executes the requested phases and repetitions exactly as specified
    via the configuration, and returns a properly structured list of TrainingMetrics.

    :param dict integrated_trainer_setup: The fixture providing the trainer and dataset.
    """
    trainer = integrated_trainer_setup["trainer"]
    dataset = integrated_trainer_setup["dataset"]

    # Overwrite the default config reps specifically for this test
    trainer.config.reps = {
        'phase_1': 2,
        'phase_2': 1,
        'phase_3': 0,  # Phase 3 is intentionally skipped
        'phase_4': 1
    }

    metrics = trainer.train(dataset=dataset, epochs=2, batch_size=2)

    # Executions per epoch = 2 + 1 + 0 + 1 = 4 phases run. Total for 2 epochs = 8 metrics.
    assert len(metrics) == 8, f"Expected 8 recorded metrics, got {len(metrics)}."

    assert isinstance(metrics[0], TrainingMetrics), "Trainer did not return TrainingMetrics objects."
    assert metrics[0].epoch == 1 and metrics[0].phase == 1 and metrics[0].rep == 1, \
        "First metric entry does not match the expected starting phase."

    # Ensure Phase 3 was entirely skipped
    phase_3_metrics = [m for m in metrics if m.phase == 3]
    assert len(phase_3_metrics) == 0, "Phase 3 was executed despite reps being set to 0."


def test_trainer_optimizer_momentum_retention(integrated_trainer_setup: dict):
    """
    Ensures that calling a phase across multiple epochs does NOT
    recreate the PyTorch optimizer, thereby securely preserving the Adam momentum states.

    :param dict integrated_trainer_setup: The fixture providing the trainer and dataset.
    """
    trainer = integrated_trainer_setup["trainer"]
    dataset = integrated_trainer_setup["dataset"]

    trainer.config.reps = {
        'phase_1': 1,
        'phase_2': 0,
        'phase_3': 0,
        'phase_4': 0
    }

    opt_id_before = id(trainer.opt_phase1.optimizer)

    _ = trainer.train(dataset=dataset, epochs=2, batch_size=2)

    opt_id_after = id(trainer.opt_phase1.optimizer)

    assert opt_id_before == opt_id_after, \
        "The Adam optimizer was recreated during training, effectively wiping its internal momentum state!"


def test_trainer_integration_overfitting(integrated_trainer_setup: dict):
    """
    Executes the full trainer pipeline across multiple epochs to prove that the gradients actively flow, weights update,
    and the total composite loss decreases. This validates the entire end-to-end forward/backward process.

    :param dict integrated_trainer_setup: The fixture providing the trainer and dataset.
    """
    trainer = integrated_trainer_setup["trainer"]
    dataset = integrated_trainer_setup["dataset"]

    trainer.config.reps = {
        'phase_1': 1,
        'phase_2': 0,
        'phase_3': 1,
        'phase_4': 0
    }

    # Run for 3 epochs to allow the optimizers to take meaningful gradient steps
    metrics = trainer.train(dataset=dataset, epochs=3, batch_size=2)

    phase1_losses = [m.total_loss for m in metrics if m.phase == 1]

    initial_loss = phase1_losses[0]
    final_loss = phase1_losses[-1]

    assert final_loss < initial_loss, \
        f"The network failed to learn and reduce error. Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}"


def test_dataloader_context_mapping_integrity(integrated_trainer_setup: dict):
    """
    Asserts that the keys extracted from the EINNDataset yield
    by the DataLoader strictly match the kwargs required to instantiate the PhaseContext,
    preventing runtime KeyErrors during batch processing.

    :param dict integrated_trainer_setup: The fixture providing the trainer and dataset.
    """
    dataset = integrated_trainer_setup["dataset"]

    # Simulate the DataLoader's output from the custom Dataset
    sample_batch = dataset[0]

    expected_keys = {'X', 'y', 't', 'aux_targets'}
    actual_keys = set(sample_batch.keys())

    assert expected_keys.issubset(actual_keys), \
        f"Dataset yield keys {actual_keys} do not match the expected PhaseContext keys {expected_keys}."


def test_trainer_early_stopping_integration(integrated_trainer_setup: dict):
    """
    Validates that the EarlyStopping mechanism correctly interrupts the training loop when the monitored metric
     (e.g., data_f in Phase 4) stops improving according to the configured patience and delta thresholds.

    :param dict integrated_trainer_setup: The fixture providing the trainer and dataset.
    """
    trainer = integrated_trainer_setup["trainer"]
    dataset = integrated_trainer_setup["dataset"]

    # Forceful Configuration for Early Stopping
    # Set an impossibly high delta so that no normal loss reduction qualifies as an improvement.
    trainer.config.early_stopping_min_delta = 1000.0
    trainer.config.early_stopping_patience = 2
    trainer.config.early_stopping_percentage = False

    # Ensure all phases run so Phase 4 generates the 'data_f' metric
    trainer.config.reps = {
        'phase_1': 1,
        'phase_2': 1,
        'phase_3': 1,
        'phase_4': 1
    }

    max_epochs = 10

    # Execute Training
    # We ask for 10 epochs, but EarlyStopping should intervene much sooner.
    metrics = trainer.train(dataset=dataset, epochs=max_epochs, batch_size=2)

    # Extract the maximum epoch actually executed
    epochs_executed = max([m.epoch for m in metrics]) if metrics else 0

    # Epoch 1: Sets the initial best loss.
    # Epoch 2: Fails to improve by 1000.0 (bad epoch count = 1).
    # Epoch 3: Fails to improve by 1000.0 (bad epoch count = 2 >= patience). Stops immediately.
    assert epochs_executed < max_epochs, \
        f"Early stopping failed! Expected to stop before {max_epochs} epochs, but ran for {epochs_executed} epochs."

    assert epochs_executed == 3, \
        f"Early stopping triggered at the wrong time." +\
        f"Expected it to stop exactly at epoch 3, but stopped at {epochs_executed}."
