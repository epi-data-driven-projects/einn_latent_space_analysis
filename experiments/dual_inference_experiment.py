"""
Experiment script comparing two differently parameterized SIR trajectories in EINN.

Generates synthetic SIR training and held-out test trajectories (high vs. low
transmission rate beta), trains a single-model SIR EINN via EINNRunner, evaluates
forecasts and Feature embeddings across four observation windows, computes joint and
per-trajectory 2D PCA projections, renders the 4x4 comparison figure, and exports
serialized `.npz` archives for figure-only regeneration and reproducibility audits.
"""
import os

import matplotlib

import numpy as np
from sklearn.decomposition import PCA
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_preprocessor import DataPreprocessor
from einn.data.einn_input_loader import EINNInputLoader
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.inference.einn_evaluator import EINNEvaluator
from einn.model.einn_builder import EINNBuilder
from einn.ode.sir_model import SIRModel
from einn.utils.experiment_utils import ExperimentUtils
from einn.visualization.dual_inference_plotter import DualInferencePlotter
from einn_runner import EINNRunner

matplotlib.use("Agg")  # Use a headless plotting backend so reports can be saved over remote SSH.

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
TARGET_IDX: int = 1  # In the three-compartment SIR state vector, index 1 denotes infectious individuals I.
NUM_STATES: int = 100
SEQ_LEN: int = 100
DEVICE: str = "cuda:1"
OUTPUT_DIR: str = "./experiment_results/sir_dual_inference"

# ======================================================================
# Keep the approved color scheme for all comparison plots.
# ======================================================================
PALETTE: dict[str, str] = {
    "ink": "#18283D",
    "muted": "#607185",
    "grid": "#D8E1EB",
    "teal": "#008E97",
    "teal_dark": "#006A76",
    "blue": "#416BA8",
    "navy": "#25415D",
    "orange": "#D78A30",
    "magenta": "#AB4C77",
    "gray": "#8192A6",
    "background_obs": "#E5F0F4",
}


def run_sir_dual_inference_experiment() -> None:
    """
    Run the SIR dual-inference training, evaluation, PCA, plotting, and `.npz` export pipeline.

    Trains an SIR EINN on synthetic trajectories sampled across beta, gamma in [0, 1], then
    evaluates two unseen SIR trajectories (beta = 0.99 vs. beta = 0.01, gamma = 0.1) across
    observation cutoffs t_end in {5, 10, 25, 50}, projects embeddings onto joint and
    single-trajectory 2D PCA bases, renders the 4x4 report, and saves reproducible archives.

    :return None: Writes training artifacts, 4x4 PNG/PDF figures, and `.npz` archives to OUTPUT_DIR.
    """
    ExperimentUtils.seed_everything(seed=42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Generate/load expanding-window SIR training data and fitted scaling statistics.
    print("--- 1. DATA PREPARATION ---")
    loader = EINNInputLoader(
        true_model=SIRModel(target_idx=TARGET_IDX),
        calibrator_model=SIRModel(target_idx=TARGET_IDX),
        # Retain the constructor argument required by the historical loader interface.
        target_idx=TARGET_IDX,
        device=DEVICE,
        cache_dir=OUTPUT_DIR,
    )

    # Sample infection and recovery rates uniformly from [0, 1] for training trajectories.
    sir_param_ranges: dict[str, tuple[float, float]] = {
        "beta": (0, 1.0),
        "gamma": (0, 1.0),
    }

    dataset, scaling_stats = loader.prepare_training_data(
        num_states=NUM_STATES,
        seq_len=SEQ_LEN,
        d_p=2,
        param_ranges=sir_param_ranges,
        initial_states=[0.99, 0.01, 0.0],
        noise_std=0.01,
        window_step=40,  # Use the original expanding-window stride of 40 timesteps.
    )

    # 2. Train single-model SIR EINN checkpoints via EINNRunner.
    print("\n--- 2. MODEL TRAINING ---")
    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=3, d_p=2, feature_n_layers=2)

    train_cfg = EINNTrainConfig(
        device=DEVICE,
        learning_rate=0.001,
        early_stopping_patience=5,
        loss_weights={
            "data_T": 10.0,
            "data_F": 10.0,
            "aux": 4.0,
            "ode_T": 8.0,
            "ode_F": 8.0,
            "ode_future_T": 0.1,
            "ode_future_F": 0.1,
            "mono": 0.0,
            "param": 0.0,
            "kd_target": 10.0,
            "kd_emb": 10.0,
            "ic": 20.0,
            "ic_F": 20.0,
        },
        reps={"phase_1": 50, "phase_2": 50, "phase_3": 50, "phase_4": 50},
    )

    sir_calibration: dict[str, float] = {"beta": 0.4, "gamma": 0.1}

    runner = EINNRunner(output_dir=OUTPUT_DIR, model_cfg=model_cfg, train_cfg=train_cfg, seed=42)
    runner.train_and_save(
        dataset=dataset,
        scaling_stats=scaling_stats,
        model_type="SIR",
        target_idx=TARGET_IDX,
        param_calibration=sir_calibration,
        epochs=10,
        batch_size=32,
    )

    # ==========================================
    # 3. GENERATE UNSEEN TEST DATA (2 DIFFERENT TRAJECTORIES)
    # ==========================================
    print("\n--- 3. GENERATING UNSEEN TEST STATES ---")
    test_correlations: dict[str, float] = {
        "mobility": -0.75,
        "temperature": -0.1,
        "testing": 0.85,
        "stringency": 0.95,
    }

    true_model = SIRModel(target_idx=TARGET_IDX)
    generator_test = SyntheticDataGenerator(true_ode_model=true_model, seq_len=SEQ_LEN)

    # Trajectory A: high transmission rate (beta = 0.99, gamma = 0.1).
    data_test_1 = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0],
        true_params=[0.99, 0.1],
        noise_std=0.01,
        correlations=test_correlations,
    )

    # Trajectory B: low transmission rate (beta = 0.01, gamma = 0.1).
    data_test_2 = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0],
        true_params=[0.01, 0.1],
        noise_std=0.01,
        correlations=test_correlations,
    )

    # Fit input feature scaler on cached training covariates x_train and transform test inputs.
    preprocessor = DataPreprocessor()
    cached_train_data = torch.load(
        os.path.join(
            OUTPUT_DIR,
            f"{NUM_STATES}_states_{SEQ_LEN}len_target_{TARGET_IDX}.pt",
        ),
        weights_only=False,
    )
    preprocessor.x_scaler.fit(np.concatenate(cached_train_data["x_train"]))
    x_scaler = preprocessor.x_scaler

    x_test_scaled_1 = x_scaler.transform(data_test_1["x_raw"])
    x_test_scaled_2 = x_scaler.transform(data_test_2["x_raw"])

    true_target_1 = data_test_1["true_latent_states"][:, TARGET_IDX]
    true_target_2 = data_test_2["true_latent_states"][:, TARGET_IDX]

    full_t_raw = torch.arange(SEQ_LEN, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(DEVICE)

    # ==========================================
    # 4. EVALUATION & COMMON PCA
    # ==========================================
    print("\n--- 4. EVALUATION & COMMON PCA ---")

    # Evaluate four fixed observation cutoffs using the trained SIR EINN checkpoints.
    phases: list[tuple[str, int]] = [
        ("1. megfigyelési ablak", 5),
        ("2. megfigyelési ablak", 10),
        ("3. megfigyelési ablak", 25),
        ("4. megfigyelési ablak", 50),
    ]

    models = EINNBuilder.build_einn(
        model_cfg,
        train_cfg,
        {},
        "SIR",
        seed=42,
    )

    models.time_module.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, "einn_time_module.pth"),
            map_location=DEVICE,
        )
    )
    models.feature_module.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, "einn_feature_module.pth"),
            map_location=DEVICE,
        )
    )
    models.output_module.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, "einn_output_module.pth"),
            map_location=DEVICE,
        )
    )

    models.ode_model = SIRModel(target_idx=TARGET_IDX)

    evaluator = EINNEvaluator(models=models, device=DEVICE)

    # ------------------------------------------
    # 4.1. Run all inference experiments
    # ------------------------------------------
    phase_results: list[dict] = []

    for phase_name, t_end in phases:
        print(f"Evaluating: {phase_name}")

        results_1 = evaluator.evaluate(
            x_test=x_test_scaled_1,
            future_t=full_t_raw,
            scaling_stats=scaling_stats,
            target_idx=TARGET_IDX,
            true_target_trajectory=true_target_1,
            t_end=t_end,
        )

        results_2 = evaluator.evaluate(
            x_test=x_test_scaled_2,
            future_t=full_t_raw,
            scaling_stats=scaling_stats,
            target_idx=TARGET_IDX,
            true_target_trajectory=true_target_2,
            t_end=t_end,
        )

        phase_results.append({
            "phase_name": phase_name,
            "t_end": t_end,
            "results_1": results_1,
            "results_2": results_2,
        })

    # ------------------------------------------
    # 4.2. Collect latent representations
    # ------------------------------------------
    latent_arrays_a = [
        phase["results_1"]["latent_feature"]
        for phase in phase_results
    ]

    latent_arrays_b = [
        phase["results_2"]["latent_feature"]
        for phase in phase_results
    ]

    all_embeddings_a = np.concatenate(latent_arrays_a, axis=0)
    all_embeddings_b = np.concatenate(latent_arrays_b, axis=0)

    all_embeddings = np.concatenate(
        [all_embeddings_a, all_embeddings_b],
        axis=0,
    )

    print("\nLatent embedding matrices:")
    print(f"A shape: {all_embeddings_a.shape}")
    print(f"B shape: {all_embeddings_b.shape}")
    print(f"Combined shape: {all_embeddings.shape}")

    # ------------------------------------------
    # 4.3. Fit the three PCA transformations
    # ------------------------------------------

    # PCA fitted jointly on pooled embeddings from trajectories A and B.
    pca_common = PCA(n_components=2)
    pca_common.fit(all_embeddings)

    # PCA fitted solely on trajectory A embeddings.
    pca_a = PCA(n_components=2)
    pca_a.fit(all_embeddings_a)

    # PCA fitted solely on trajectory B embeddings.
    pca_b = PCA(n_components=2)
    pca_b.fit(all_embeddings_b)

    # Full-spectrum PCA on combined embeddings to inspect cumulative explained variance.
    pca_variance = PCA()
    pca_variance.fit(all_embeddings)

    cum_variance = np.cumsum(
        pca_variance.explained_variance_ratio_
    )

    n_comp_95 = int(np.argmax(cum_variance >= 0.95) + 1)

    print(
        f"Dimensions required for 95% variance "
        f"(common PCA): {n_comp_95}"
    )

    # ------------------------------------------
    # 4.4. Apply all PCA transformations
    # ------------------------------------------
    for phase in phase_results:
        emb_a = phase["results_1"]["latent_feature"]
        emb_b = phase["results_2"]["latent_feature"]

        # 1. Common PCA applied to both trajectories.
        phase["pca_common_1"] = pca_common.transform(emb_a)
        phase["pca_common_2"] = pca_common.transform(emb_b)

        # 2. Trajectory-A PCA applied to both trajectories.
        phase["pca_a_1"] = pca_a.transform(emb_a)
        phase["pca_a_2"] = pca_a.transform(emb_b)

        # 3. Trajectory-B PCA applied to both trajectories.
        phase["pca_b_1"] = pca_b.transform(emb_a)
        phase["pca_b_2"] = pca_b.transform(emb_b)

    print("\nPCA transformations completed successfully.")

    # ==========================================
    # 5. PLOTTING & SERIALIZATION
    # ==========================================
    print("\n--- 5. PLOTTING ---")
    png_path, pdf_path = plot_dual_inference_results(
        phase_results=phase_results,
        true_target_1=true_target_1,
        true_target_2=true_target_2,
        output_dir=OUTPUT_DIR,
        embedding_dim=model_cfg.d_e,
    )
    print(f"\nÁbra mentve (PNG): {png_path}\nÁbra mentve (PDF): {pdf_path}")

    # Save packed PCA projections and reference targets for offline chart regeneration.
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "dual_inference_saved_arrays.npz"),
        observation_windows=np.asarray([p["t_end"] for p in phase_results]),
        true_target_1=np.asarray(true_target_1),
        true_target_2=np.asarray(true_target_2),
        embedding_dim=np.asarray(model_cfg.d_e),
        **{
            f"{key}_{index}": np.asarray(phase[key])
            for index, phase in enumerate(phase_results)
            for key in ("pca_common_1", "pca_common_2", "pca_a_1",
                        "pca_a_2", "pca_b_1", "pca_b_2")
        },
    )

    # Save per-window evaluation metrics and arrays for both test trajectories.
    for index, phase in enumerate(phase_results):
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"dual_inference_window_{index}.npz"),
            **{
                f"a_{key}": np.asarray(value)
                for key, value in phase["results_1"].items()
                if isinstance(value, (np.ndarray, list, tuple, int, float))
            },
            **{
                f"b_{key}": np.asarray(value)
                for key, value in phase["results_2"].items()
                if isinstance(value, (np.ndarray, list, tuple, int, float))
            },
        )

    # Save raw/scaled test realizations, true parameters, and fitted scaler parameters for audits.
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "dual_inference_test_inputs.npz"),
        **{
            f"high_beta_{name}": value
            for name, value in data_test_1.items()
            if isinstance(value, np.ndarray)
        },
        **{
            f"low_beta_{name}": value
            for name, value in data_test_2.items()
            if isinstance(value, np.ndarray)
        },
        high_beta_scaled_inputs=x_test_scaled_1,
        low_beta_scaled_inputs=x_test_scaled_2,
        high_beta_parameters=np.asarray([0.99, 0.1]),
        low_beta_parameters=np.asarray([0.01, 0.1]),
        input_scaler_mean=x_scaler.mean_,
        input_scaler_scale=x_scaler.scale_,
    )


# Reuse the exact original dual-inference plotting implementation.
plot_dual_inference_results = DualInferencePlotter().plot_dual_inference_results
