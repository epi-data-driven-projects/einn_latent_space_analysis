from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.evaluation.generalization_evaluator import (
    EINNGeneralizationEvaluator,
    GeneralizationEvaluation,
)
from experiment.einn_runner import EINNRunner, EINNRunResult
from einn.ode.seirm_model import SEIRMModel
from einn.visualization.experiment_plotter import EINNExperimentPlotter


@dataclass
class GeneralizationExperimentResult:
    """Combined training and held-out generalization result."""

    run_result: EINNRunResult
    evaluation: GeneralizationEvaluation
    output_dir: str


class EINNGeneralizationExperiment:
    """Reproduce the former SEIRM generalization experiment with the new runner.

    Training and model hyperparameters follow the supplied legacy script.
    Only the expanding-window settings are adapted to the new window API:
    20-day minimum prefix, 56-step forecast horizon and stride 20. This gives
    the same ten prefix endpoints (20, 40, ..., 200) per training trajectory
    as the old ExpandingWindowDataset, while using target-aligned windows.
    """

    TRUE_INITIAL_STATES = [0.99, 0.01, 0.0, 0.0, 0.0]
    ASSUMED_INITIAL_STATES = [0.99, 0.01, 0.0, 0.0, 0.0]

    TRUE_PARAMETER_RANGES = {
        "beta": (0.30, 0.60),
        "alpha": (0.10, 0.30),
        "gamma": (0.05, 0.15),
        "mu": (0.01, 0.08),
    }
    CORRELATION_RANGES = {
        "mobility": (-0.90, -0.40),
        "temperature": (-0.40, 0.10),
        "testing": (0.50, 0.90),
        "stringency": (0.70, 0.95),
    }
    TEST_PARAMETERS = {
        "beta": 0.45,
        "alpha": 0.15,
        "gamma": 0.12,
        "mu": 0.04,
    }
    TEST_CORRELATIONS = {
        "mobility": -0.75,
        "temperature": -0.10,
        "testing": 0.85,
        "stringency": 0.95,
    }
    ODE_INITIAL_PARAMETERS = {
        "beta": 0.40,
        "alpha": 0.20,
        "gamma": 0.10,
        "mu": 0.05,
    }

    def __init__(
        self,
        output_dir: str | Path = "./experiment_results/generalization",
        target_idx: int = 2,
        num_states: int = 20,
        seq_len: int = 200,
        device: Optional[str] = None,
        seed: int = 42,
        min_sequence_length: int = 20,
        forecast_horizon: int = 56,
        window_stride: int = 20,
        calibration_epochs: int = 300,
        calibration_lr: float = 0.001,
        training_epochs: int = 20,
        batch_size: int = 32,
    ) -> None:
        if target_idx not in (1, 2, 4):
            raise ValueError("The legacy experiment uses TARGET_IDX in {1, 2, 4}.")
        if num_states < 1:
            raise ValueError("num_states must be at least 1.")
        if seq_len <= 150:
            raise ValueError("seq_len must exceed 150 for the original day-150 diagnostic.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_idx = target_idx
        self.num_states = num_states
        self.seq_len = seq_len
        self.seed = seed
        self.min_sequence_length = min_sequence_length
        self.forecast_horizon = forecast_horizon
        self.window_stride = window_stride
        self.calibration_epochs = calibration_epochs
        self.calibration_lr = calibration_lr
        self.training_epochs = training_epochs
        self.batch_size = batch_size

        requested_device = device or (
            "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else "cuda:0" if torch.cuda.is_available()
            else "cpu"
        )
        self.device = self._available_device(requested_device)

        self.model_config = EINNModelConfig(
            d_x=5,
            d_e=20,
            d_s=5,
            d_p=4,
            feature_n_layers=2,
        )
        self.train_config = EINNTrainConfig(
            device=self.device,
            learning_rate=0.001,
            early_stopping_patience=5,
            loss_weights={
                "data_T": 10.0,
                "data_F": 10.0,
                "aux": 4.0,
                "ode_T": 8.0,
                "ode_F": 8.0,
                "ode_future_T": 1.0,
                "ode_future_F": 1.0,
                "mono": 3.0,
                "param": 0.1,
                "kd_target": 2.0,
                "kd_emb": 2.0,
                "ic": 10.0,
                "ic_F": 10.0,
            },
        )
        self.train_config.reps = {
            "phase_1": 100,
            "phase_2": 100,
            "phase_3": 100,
            "phase_4": 100,
        }

        true_model = SEIRMModel(target_idx=target_idx)
        assumed_model = SEIRMModel(target_idx=target_idx)

        self.runner = EINNRunner(
            output_dir=self.output_dir,
            true_model=true_model,
            assumed_model=assumed_model,
            model_config=self.model_config,
            train_config=self.train_config,
            n_train_series=num_states,
            seq_len=seq_len,
            min_sequence_length=min_sequence_length,
            forecast_horizon=forecast_horizon,
            stride=window_stride,
            true_initial_states=self.TRUE_INITIAL_STATES,
            assumed_initial_states=self.ASSUMED_INITIAL_STATES,
            true_parameter_ranges=self.TRUE_PARAMETER_RANGES,
            parameter_names=["beta", "alpha", "gamma", "mu"],
            correlation_ranges=self.CORRELATION_RANGES,
            noise_std=0.02,
            calibration_epochs=calibration_epochs,
            calibration_lr=calibration_lr,
            ode_initial_parameters=self.ODE_INITIAL_PARAMETERS,
            seed=seed,
            calibration_device=self.device,
        )

        self.evaluator = EINNGeneralizationEvaluator(
            device=self.runner.train_config.device,
            target_idx=target_idx,
            tsne_perplexity=10.0,
            random_state=seed,
        )
        self.plotter = EINNExperimentPlotter(self.output_dir)

    @staticmethod
    def _available_device(requested: str) -> str:
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                return "cpu"
            if ":" in requested:
                index = int(requested.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    return "cuda:0"
        return requested

    def run(self, force_regenerate: bool = False) -> GeneralizationExperimentResult:
        """Run training, held-out inference, evaluation and plotting."""
        run_result = self.runner.run_experiment(
            epochs=self.training_epochs,
            batch_size=self.batch_size,
            force_regenerate=force_regenerate,
            test_params=self.TEST_PARAMETERS,
            test_correlations=self.TEST_CORRELATIONS,
            test_noise_std=0.01,
        )

        evaluation = self.evaluator.evaluate(
            models=run_result.models,
            test_series=run_result.test_series,
            x_scaler=run_result.preprocessor.x_scaler,
            scaling_stats=run_result.scaling_stats,
        )

        # Keep the runner's result object consistent with the full-history
        # evaluation prediction, and refresh the held-out archive accordingly.
        run_result.predicted_states = evaluation.full_prediction
        self.plotter.plot_loss_curves(run_result.metrics)
        self.plotter.plot_true_vs_predicted(
            true_states=evaluation.true_states,
            predicted_states=evaluation.full_prediction,
        )
        grid_path = self.plotter.plot_generalization_grid(
            evaluation=evaluation,
            d_e=self.model_config.d_e,
            filename=f"generalization_full_analysis_target_{self.target_idx}.png",
        )
        metrics_path = self.plotter.save_evaluation(evaluation)

        # Persist the exact test series used for all evaluation plots.
        import numpy as np

        np.savez_compressed(
            self.output_dir / "held_out_prediction.npz",
            x_raw=run_result.test_series.x_raw,
            y_raw=run_result.test_series.y_raw,
            true_latent_states=run_result.test_series.true_latent_states,
            predicted_states=evaluation.full_prediction,
            true_params=run_result.test_series.true_params,
            correlations=run_result.test_series.correlations,
        )

        print(f"Generalization grid saved to: {grid_path}")
        print(f"Generalization metrics saved to: {metrics_path}")
        return GeneralizationExperimentResult(
            run_result=run_result,
            evaluation=evaluation,
            output_dir=str(self.output_dir),
        )
