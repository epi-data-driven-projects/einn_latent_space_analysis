import json

import torch

from einn.config.einn_config import EINNConfig
from einn.data.data_downloader import DataDownloader
from einn.data.data_preprocessor import DataPreprocessor
from einn.data.einn_dataset import EINNDataset
from einn.model.einn_builder import EINNBuilder
from einn.training.einn_trainer import EINNTrainer
from einn.inference.inference_engine import InferenceEngine
from einn.inference.training_post_processor import TrainingPostProcessor


def run() -> None:
    """
    Main entry point for the EINN workflow.
    Executes data loading, preprocessing, model building, training, inference, and saving.
    """
    print("--- EINN ALGORITHM STARTING ---")

    # Initialize configuration (detects GPU automatically)
    config = EINNConfig()
    device = torch.device(config.device)
    print(f"Active Compute Device: {device}")

    # 1. Obtain raw data
    downloader = DataDownloader()
    X, y, t = downloader.download_data()

    # Ensure tensors have a batch dimension: [1, Seq_len, Features]
    if X.dim() == 2: X = X.unsqueeze(0)
    if y.dim() == 2: y = y.unsqueeze(0)
    if t.dim() == 2: t = t.unsqueeze(0)

    # Calculate sequence length
    seq_len = X.shape[1]

    # Load JSON calibration if available
    param_calibration = {}
    try:
        with open("data/analytical/seirm-t-rmse-calibration.json", "r") as f:
            param_calibration = json.load(f)
        print("Loaded initial ODE calibration from JSON.")
    except FileNotFoundError:
        print("Calibration JSON not found. Default parameters will be used.")

    # 2. Build the models based on config and move them to target device
    models = EINNBuilder.build_einn(config, seq_len=seq_len, param_calibration=param_calibration)

    # 3. Preprocess and scale data, generate aux_targets
    preprocessor = DataPreprocessor(device=config.device)
    X_scaled, y_scaled, t_scaled = preprocessor.scale_data(X, y, t)
    aux_targets = preprocessor.generate_aux_targets(t_scaled, models.ode_model, d_s=config.d_s)

    # 4. Initialize PyTorch Dataset
    dataset = EINNDataset(
        x=X_scaled,
        y=y_scaled,
        t=t_scaled,
        aux_targets=aux_targets,
        device=config.device
    )

    # 5. Run the Training process
    trainer = EINNTrainer(models=models, scalers=preprocessor.scalers, config=config)

    # Store returned metrics for later saving
    metrics_list = trainer.train(dataset, config)

    # 6. Future predictions (Inference)
    inference = InferenceEngine(models, device=config.device)

    # Generate future time vector maintaining the scaled time step (dt)
    dt = t_scaled[0, 1, 0] - t_scaled[0, 0, 0] if t_scaled.shape[1] > 1 else 0.1
    last_t = t_scaled[0, -1, 0]

    # Create future_t directly on the configured device
    future_t = torch.tensor(
        [last_t + (i + 1) * dt for i in range(config.future_steps)],
        device=device
    ).view(1, config.future_steps, 1).float()

    # Predict future normalized states
    s_future_scaled = inference.predict(X_scaled, future_t)

    # Revert normalization to physical scale
    s_future_real = preprocessor.inverse_scale_predictions(s_future_scaled)

    # 7. Save Results (Weights, Config, Metrics, Predictions)
    post_processor = TrainingPostProcessor()

    # Using generic state names if you swap SEIRM to SIR
    state_names = [f"State_{i}" for i in range(config.d_s)]

    post_processor.process_and_save(
        models=models,
        config=config,
        metrics_list=metrics_list,
        predictions=s_future_real,
        state_names=state_names
    )

    print("--- RUN SUCCESSFULLY COMPLETED ---")


if __name__ == "__main__":
    run()
