import pytest
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.inference.inference_engine import InferenceEngine
from einn.model.einn_builder import EINNBuilder


@pytest.fixture
def integrated_inference_setup() -> dict:
    """
    Fixture providing initialized components for inference tests.

    :return dict: A dictionary containing the inference engine and its models.
    """
    model_config = EINNModelConfig(
        d_x=5,
        d_e=10,
        d_s=5,
        d_p=4,
        feature_n_layers=1,
    )
    train_config = EINNTrainConfig(device="cpu")

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
        target_idx=1,
    )

    return {
        "engine": InferenceEngine(models=models),
        "models": models,
    }


@pytest.fixture
def dummy_scaling_stats() -> DataScalingStats:
    """
    Fixture providing neutral z-score scaling and time bounds matching future_t.

    :return DataScalingStats: Initialized dummy scaling statistics.
    """
    return DataScalingStats(
        aux_mean=torch.zeros(5),
        aux_std=torch.ones(5),
        t_min=torch.tensor(0.0),
        t_max=torch.tensor(15.0),
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
    )


@pytest.fixture
def dummy_inference_inputs(
    dummy_scaling_stats: DataScalingStats,
) -> dict:
    """
    Fixture providing the input tensors required by predict_future().

    The mask convention is 0 for observed time points and 1 for padding.

    :param DataScalingStats dummy_scaling_stats: Neutral z-score scaling statistics.
    :return dict: Keyword arguments for predict_future().
    """
    batch_size = 1
    seq_len = 10
    d_x = 5
    future_steps = 5

    return {
        "historical_x": torch.rand(size=(batch_size, seq_len, d_x)),
        "historical_x_mask": torch.zeros(size=(batch_size, seq_len)),
        "future_t": torch.linspace(11.0, 15.0, future_steps).view(
            1, future_steps, 1
        ),
        "scaling_stats": dummy_scaling_stats,
    }


def test_predict_future_routing_and_shapes(
    integrated_inference_setup: dict,
    dummy_inference_inputs: dict,
) -> None:
    """
    Validate prediction shape, finite values, and disabled gradient tracking.
    """
    engine = integrated_inference_setup["engine"]
    predictions = engine.predict_future(**dummy_inference_inputs)

    assert predictions.shape == (1, 5, 5), (
        f"Expected prediction shape (1, 5, 5), got {predictions.shape}."
    )
    assert torch.isfinite(predictions).all(), (
        "Inference returned non-finite values."
    )
    assert not predictions.requires_grad, (
        "Gradients are being tracked during inference."
    )


def test_predict_future_inverse_scaling(
    integrated_inference_setup: dict,
    dummy_inference_inputs: dict,
) -> None:
    """
    Verify inverse z-score scaling while keeping the time scaling unchanged.
    """
    engine = integrated_inference_setup["engine"]

    raw_predictions = engine.predict_future(**dummy_inference_inputs)
    assert torch.isfinite(raw_predictions).all(), (
        "Raw inference returned non-finite values."
    )

    real_z_mean = torch.tensor(
        [100.0, 50.0, 20.0, 10.0, 5.0],
        dtype=torch.float32,
    )
    real_z_std = torch.tensor(
        [15.0, 10.0, 5.0, 2.0, 1.0],
        dtype=torch.float32,
    )

    real_scaling_stats = DataScalingStats(
        aux_mean=real_z_mean,
        aux_std=real_z_std,
        t_min=torch.tensor(0.0),
        t_max=torch.tensor(15.0),
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
    )

    scaled_inputs = dummy_inference_inputs.copy()
    scaled_inputs["scaling_stats"] = real_scaling_stats

    scaled_predictions = engine.predict_future(**scaled_inputs)
    assert torch.isfinite(scaled_predictions).all(), (
        "Inference with non-neutral z-score scaling returned non-finite values."
    )

    expected_predictions = raw_predictions * real_z_std + real_z_mean

    assert torch.allclose(
        scaled_predictions,
        expected_predictions,
        atol=1e-5,
        rtol=1e-5,
    ), "Inverse z-score scaling did not transform the predictions as expected."


def test_eval_mode_activation(
    integrated_inference_setup: dict,
    dummy_inference_inputs: dict,
) -> None:
    """
    Verify that inference switches all model modules to evaluation mode.
    """
    engine = integrated_inference_setup["engine"]
    models = integrated_inference_setup["models"]

    models.time_module.train()
    models.feature_module.train()
    models.output_module.train()

    engine.predict_future(**dummy_inference_inputs)

    assert not models.time_module.training, (
        "Time module was not switched to eval mode."
    )
    assert not models.feature_module.training, (
        "Feature module was not switched to eval mode."
    )
    assert not models.output_module.training, (
        "Output module was not switched to eval mode."
    )


def test_inference_determinism_and_dropout_disabled(
    integrated_inference_setup: dict,
    dummy_inference_inputs: dict,
) -> None:
    """
    Verify that repeated inference with identical inputs returns the same result.
    """
    engine = integrated_inference_setup["engine"]

    pred_1 = engine.predict_future(**dummy_inference_inputs)
    pred_2 = engine.predict_future(**dummy_inference_inputs)

    assert torch.isfinite(pred_1).all() and torch.isfinite(pred_2).all(), (
        "Inference returned non-finite values."
    )
    assert torch.allclose(pred_1, pred_2, atol=1e-7, rtol=0.0), (
        "Inference is non-deterministic; a model component may still be active "
        "in training mode."
    )


def test_inference_input_sensitivity(
    integrated_inference_setup: dict,
    dummy_inference_inputs: dict,
) -> None:
    """
    Verify that predictions respond to changes in historical input features.
    """
    engine = integrated_inference_setup["engine"]

    pred_baseline = engine.predict_future(**dummy_inference_inputs)

    mutated_inputs = dummy_inference_inputs.copy()
    mutated_inputs["historical_x"] = (
        mutated_inputs["historical_x"] + 5.0
    )

    pred_mutated = engine.predict_future(**mutated_inputs)

    assert torch.isfinite(pred_baseline).all() and torch.isfinite(pred_mutated).all(), (
        "Inference returned non-finite values."
    )
    assert not torch.allclose(
        pred_baseline,
        pred_mutated,
        atol=1e-3,
        rtol=1e-3,
    ), "Predictions did not change after modifying the historical inputs."


def test_inference_batch_independence(
    integrated_inference_setup: dict,
    dummy_inference_inputs: dict,
) -> None:
    """
    Verify that one sample's prediction is independent of other batch samples.
    """
    engine = integrated_inference_setup["engine"]
    inputs_a = dummy_inference_inputs

    inputs_b = {
        "historical_x": torch.rand_like(inputs_a["historical_x"]),
        "historical_x_mask": torch.zeros_like(
            inputs_a["historical_x_mask"]
        ),
        "future_t": inputs_a["future_t"].clone(),
        "scaling_stats": inputs_a["scaling_stats"],
    }

    batch_inputs = {
        "historical_x": torch.cat(
            [inputs_a["historical_x"], inputs_b["historical_x"]],
            dim=0,
        ),
        "historical_x_mask": torch.cat(
            [
                inputs_a["historical_x_mask"],
                inputs_b["historical_x_mask"],
            ],
            dim=0,
        ),
        "future_t": torch.cat(
            [inputs_a["future_t"], inputs_b["future_t"]],
            dim=0,
        ),
        "scaling_stats": inputs_a["scaling_stats"],
    }

    joint_predictions = engine.predict_future(**batch_inputs)
    independent_prediction_a = engine.predict_future(**inputs_a)

    assert torch.isfinite(joint_predictions).all(), (
        "Batched inference returned non-finite values."
    )
    assert torch.isfinite(independent_prediction_a).all(), (
        "Single-sample inference returned non-finite values."
    )
    assert torch.allclose(
        joint_predictions[0:1],
        independent_prediction_a,
        atol=1e-6,
        rtol=1e-6,
    ), (
        "Batch independence failed: predictions differ between joint and "
        "separate inference."
    )
