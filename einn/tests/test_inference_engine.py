import pytest
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.model.einn_builder import EINNBuilder
from einn.inference.inference_engine import InferenceEngine


@pytest.fixture
def integrated_inference_setup() -> dict:
    """
    Fixture providing all initialized, real components needed to test the InferenceEngine.

    :return dict: A dictionary containing the setup for inference, holding the engine and the built models.
    """
    model_config = EINNModelConfig(d_x=5, d_e=10, d_s=5, d_p=4, feature_n_layers=1)
    train_config = EINNTrainConfig(device='cpu')

    models = EINNBuilder.build_einn(
        model_config=model_config,
        train_config=train_config,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
        model_type="SEIRM",
        seed=42
    )

    return {
        "engine": InferenceEngine(models=models),
        "models": models
    }


@pytest.fixture
def dummy_scaling_stats() -> DataScalingStats:
    """
    Fixture providing a neutral DataScalingStats object.
    Mean=0 and Std=1 means the inverse scaling will effectively be an identity operation
    (outputs remain mathematically identical to the raw neural network output).
    Time bounds are strictly 0 and 1.
    """
    return DataScalingStats(
        aux_mean=torch.zeros(5),
        aux_std=torch.ones(5),
        t_min=torch.tensor(0.0),
        t_max=torch.tensor(1.0),
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0]
    )


@pytest.fixture
def dummy_inference_inputs(dummy_scaling_stats: DataScalingStats) -> dict:
    """
    Fixture providing standard baseline tensor inputs for inference testing.
    This eliminates code repetition across tests.

    :param DataScalingStats dummy_scaling_stats: The neutral scaling stats.
    :return dict: a kwargs dictionary containing the required inputs for predict_future().
    """
    batch_size, seq_len, d_x, future_steps = 1, 10, 5, 5

    return {
        "historical_x": torch.rand(size=(batch_size, seq_len, d_x)),
        "historical_x_mask": torch.ones(size=(batch_size, seq_len)),
        "future_t": torch.linspace(11, 15, future_steps).unsqueeze(0).unsqueeze(2),
        "scaling_stats": dummy_scaling_stats
    }


def test_predict_future_routing_and_shapes(integrated_inference_setup: dict, dummy_inference_inputs: dict):
    """
    Validates that the engine correctly processes genuine tensors and returns the expected shape.
    It also strictly ensures that gradient tracking is disabled to prevent memory leaks during evaluation.
    """
    engine = integrated_inference_setup["engine"]

    # Execute the core forward pass
    predictions = engine.predict_future(**dummy_inference_inputs)

    # Validate architectural dimensions: Batch=1, Future_steps=5, d_s=5 (SEIRM compartments)
    assert predictions.shape == (1, 5, 5), \
        f"Expected prediction shape (1, 5, 5), but got {predictions.shape}."

    # Validate memory safety: Inference should NEVER build a computation graph
    assert not predictions.requires_grad, \
        "Gradients are being tracked during inference! This will cause massive memory leaks."


def test_predict_future_inverse_scaling(integrated_inference_setup: dict, dummy_inference_inputs: dict):
    """
    Ensures the Z-score inverse scaling is mathematically accurate by comparing
    predictions run through a "neutral" scaler versus a "real" scaler.
    """
    engine = integrated_inference_setup["engine"]

    # Obtain raw predictions using the neutral identity scaler (mean=0, std=1)
    raw_predictions = engine.predict_future(**dummy_inference_inputs)

    # Define realistic, non-zero scaling statistics
    real_z_mean = torch.tensor([100.0, 50.0, 20.0, 10.0, 5.0], dtype=torch.float32)
    real_z_std = torch.tensor([15.0, 10.0, 5.0, 2.0, 1.0], dtype=torch.float32)

    real_scaling_stats = DataScalingStats(
        aux_mean=real_z_mean,
        aux_std=real_z_std,
        t_min=torch.tensor(0.0),
        t_max=torch.tensor(1.0),
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0]
    )

    # Swap the neutral scaler for the real scaler in the input dictionary
    scaled_inputs = dummy_inference_inputs.copy()
    scaled_inputs["scaling_stats"] = real_scaling_stats

    # Execute prediction with active scaling
    scaled_predictions = engine.predict_future(**scaled_inputs)

    # Mathematical expectation: output = (raw_output * std) + mean
    expected_output = (raw_predictions * real_z_std) + real_z_mean

    assert torch.allclose(scaled_predictions, expected_output, atol=1e-5), \
        "The inverse Z-score scaling logic failed to mathematically transform the predictions."


def test_eval_mode_activation(integrated_inference_setup: dict, dummy_inference_inputs: dict):
    """
    Verifies that predict_future actively locks all actual neural network modules into evaluation mode.
    """
    engine = integrated_inference_setup["engine"]
    models = integrated_inference_setup["models"]

    models.time_module.train()
    models.feature_module.train()
    models.output_module.train()

    engine.predict_future(**dummy_inference_inputs)

    assert not models.time_module.training, "Time module was not switched to eval mode."
    assert not models.feature_module.training, "Feature module was not switched to eval mode."
    assert not models.output_module.training, "Output module was not switched to eval mode."


def test_inference_determinism_and_dropout_disabled(integrated_inference_setup: dict, dummy_inference_inputs: dict):
    """
    Proves that running inference multiple times with the exact same inputs yields the exact same outputs.
    """
    engine = integrated_inference_setup["engine"]

    pred_1 = engine.predict_future(**dummy_inference_inputs)
    pred_2 = engine.predict_future(**dummy_inference_inputs)

    assert torch.equal(pred_1, pred_2), \
        "Inference is non-deterministic! Model components (like Dropout) might still be active."


def test_inference_input_sensitivity(integrated_inference_setup: dict, dummy_inference_inputs: dict):
    """
    Proves that the neural network's predictions are dynamically driven by the input features.
    """
    engine = integrated_inference_setup["engine"]

    pred_baseline = engine.predict_future(**dummy_inference_inputs)

    # Copying inputs, and changing X vector
    mutated_inputs = dummy_inference_inputs.copy()
    mutated_inputs["historical_x"] = mutated_inputs["historical_x"] + 5.0

    pred_mutated = engine.predict_future(**mutated_inputs)

    assert not torch.allclose(pred_baseline, pred_mutated, atol=1e-3), \
        "The model's predictions did not change despite a massive shift in historical inputs!"


def test_inference_batch_independence(integrated_inference_setup: dict, dummy_inference_inputs: dict):
    """
    Ensures there is no "cross-talk" between different items in the same batch.
    """
    engine = integrated_inference_setup["engine"]

    # region "A" = original dummy input
    inputs_a = dummy_inference_inputs

    # region "B" = a new, random input
    inputs_b = {
        "historical_x": torch.rand_like(inputs_a["historical_x"]),
        "historical_x_mask": torch.ones_like(inputs_a["historical_x_mask"]),
        "future_t": inputs_a["future_t"].clone(),
        "scaling_stats": inputs_a["scaling_stats"]  # Same scaler
    }

    # Creating shared batch, with concatenation (batch size: 1 + 1 = 2)
    batch_inputs = {
        "historical_x": torch.cat(tensors=[inputs_a["historical_x"], inputs_b["historical_x"]], dim=0),
        "historical_x_mask": torch.cat(tensors=[inputs_a["historical_x_mask"], inputs_b["historical_x_mask"]], dim=0),
        "future_t": torch.cat(tensors=[inputs_a["future_t"], inputs_b["future_t"]], dim=0),
        "scaling_stats": inputs_a["scaling_stats"]
    }

    joint_predictions = engine.predict_future(**batch_inputs)
    independent_prediction_a = engine.predict_future(**inputs_a)

    # Joint prediction's 0th index (region "A") should be equal to the independent prediction
    assert torch.allclose(joint_predictions[0:1], independent_prediction_a, atol=1e-6), \
        "Batch independence failed! The network is mixing signals between different batch items."
