"""Minimal entry point for the user's existing EINN project."""

import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn_runner import EINNRunner
from einn.ode.seirm_model import SEIRMModel


if __name__ == "__main__":
    true_model = SEIRMModel(target_idx=2)
    assumed_model = SEIRMModel(target_idx=2)

    model_config = EINNModelConfig(
        d_x=5,
        d_e=20,
        d_s=5,
        d_p=4,
        feature_n_layers=2,
    )

    train_config = EINNTrainConfig(
        device="cuda:1" if torch.cuda.is_available() else "cpu",
        learning_rate=0.001,
        early_stopping_patience=10,
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

    # Start with a small run. Increase these only after the pipeline passes.
    train_config.reps = {
        "phase_1": 1,
        "phase_2": 1,
        "phase_3": 1,
        "phase_4": 1,
    }

    runner = EINNRunner(
        output_dir="./experiment_results/seirm_run",
        true_model=true_model,
        assumed_model=assumed_model,
        model_config=model_config,
        train_config=train_config,
        n_train_series=2,
        seq_len=80,
        min_sequence_length=10,
        forecast_horizon=8,
        stride=2,
        true_initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
        assumed_initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
        noise_std=0.02,
        calibration_epochs=3,
        calibration_lr=0.01,
        seed=42,
    )

    result = runner.run_experiment(epochs=2, batch_size=4)
    print("Prediction shape:", result.predicted_states.shape)
    print("Saved to:", result.output_dir)
