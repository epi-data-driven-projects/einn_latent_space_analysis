"""
Two-model EINN experiment script with per-window 4x4 generalization and joint Hausdorff reports.

Configures and runs a true-model-only two-model EINN experiment (SIR/SIS or SIR/SEIR)
with SupConLoss, saves preprocessed training and held-out trajectories plus scaling
parameters for reproducibility audits, trains and persists checkpoints/metrics via
EINNTrainer and TrainingPostProcessor, and delegates 4x4 and joint latent-space report
generation to FeatureOnlyEvaluator.
"""
import csv
import logging
import os
from typing import Any

import matplotlib
import numpy as np
from scipy.spatial import cKDTree
import torch

from einn_multimodel.config.einn_model_config import EINNModelConfig
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.data.multi_model_input_loader import MultiModelInputLoader
from einn_multimodel.inference.feature_only_evaluator import FeatureOnlyEvaluator
from einn_multimodel.inference.training_post_processor import TrainingPostProcessor
from einn_multimodel.model.einn_builder import EINNBuilder
from einn_multimodel.training.einn_trainer import EINNTrainer

matplotlib.use("Agg")

# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

SEED: int = 42
MODEL_NAMES: tuple[str, str] = ("SIR", "SIS")   # Supported pairs: ("SIR", "SIS") or ("SIR", "SEIR").
TRAIN_STATES_PER_MODEL: int = 100
TEST_STATES_PER_MODEL: int = 20
SEQ_LEN: int = 200
NOISE_STD: float = 0.01
EPOCHS: int = 5
BATCH_SIZE: int = 32
DEVICE: str = "cuda:1"

EMBEDDING_DIM: int = 20                 # Latent embedding dimension d_e (e.g. 5 or 20).
OBSERVATION_WINDOWS: tuple[int, ...] = (10, 18, 40, 80)
PCA_COMPONENTS_TO_SHOW: int = 8

NORMALIZATION_MODES: tuple[bool, ...] = (False,)
FEATURE_LABEL_CONDITIONING_MODES: tuple[bool, ...] = (True,)
OUTPUT_ROOT: str = f"./outputs/{'_'.join(n.lower() for n in MODEL_NAMES)}_generalization_4x4"

PARAM_CALIBRATION: dict[str, dict[str, float]] = {
    "SIR": {"beta": 0.4, "gamma": 0.1},
    "SIS": {"beta": 0.4, "gamma": 0.1},
    "SEIR": {"beta": 0.4, "alpha": 0.2, "gamma": 0.1},
}
MODEL_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "SIR": {"beta": (0.0, 1.0), "gamma": (0.0, 1.0)},
    "SIS": {"beta": (0.0, 1.0), "gamma": (0.0, 1.0)},
    "SEIR": {"beta": (0.0, 1.0), "alpha": (0.0, 1.0), "gamma": (0.0, 1.0)},
}

LOSS_WEIGHTS: dict[str, float] = {
    "data_T": 4.0,
    "data_F": 4.0,
    "aux": 4.0,
    "ode_T": 10.0,
    "ode_F": 10.0,
    "ode_future_T": 10.0,
    "ode_future_F": 10.0,
    "mono": 1.0,
    "param": 0.0,
    "kd_target": 8.0,
    "kd_emb": 8.0,
    "ic": 10.0,
    "ic_F": 10.0,
    "supcon": 5.0,
}


# =====================================================================
# VISUAL STYLE
# =====================================================================

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
PNG_DPI: int = 230


# =====================================================================
# METRIC HELPERS
# =====================================================================

def paired_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute row-wise cosine similarity between aligned embedding trajectories a and b.

    Returns NaN at positions where either vector norm is at or below 1e-12 to avoid
    division-by-zero instabilities.

    :param np.ndarray a: First embedding array of shape (T, D).
    :param np.ndarray b: Second embedding array of shape (T, D) aligned row-wise with a.
    :return np.ndarray: 1D cosine similarity array of shape (T,).
    """
    numer = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numer,
        denom,
        out=np.full(a.shape[0], np.nan, dtype=float),
        where=denom > 1e-12,
    )


def hausdorff_statistics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """
    Compute symmetric, directed, 95th-percentile, and mean nearest-neighbor Hausdorff metrics.

    Evaluates Euclidean nearest-neighbor distances in the original embedding space R^D
    using `scipy.spatial.cKDTree`.

    :param np.ndarray a: Point cloud array of shape (N_a, D) for model class A.
    :param np.ndarray b: Point cloud array of shape (N_b, D) for model class B.
    :return dict[str, float]: Dictionary with keys 'hausdorff', 'directed_a_to_b',
        'directed_b_to_a', 'hausdorff_p95', 'mean_nn_a_to_b', and 'mean_nn_b_to_a'.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    distance_a_to_b = cKDTree(b_arr).query(a_arr, k=1, workers=1)[0]
    distance_b_to_a = cKDTree(a_arr).query(b_arr, k=1, workers=1)[0]
    directed_ab = float(distance_a_to_b.max())
    directed_ba = float(distance_b_to_a.max())
    return {
        "hausdorff": max(directed_ab, directed_ba),
        "directed_a_to_b": directed_ab,
        "directed_b_to_a": directed_ba,
        "hausdorff_p95": float(max(
            np.percentile(distance_a_to_b, 95),
            np.percentile(distance_b_to_a, 95),
        )),
        "mean_nn_a_to_b": float(distance_a_to_b.mean()),
        "mean_nn_b_to_a": float(distance_b_to_a.mean()),
    }


def extract_embeddings(output: object) -> torch.Tensor:
    """
    Extract a latent embedding tensor from a FeatureModuleOutputs object or raw Tensor.

    :param object output: FeatureModule forward return value or direct torch.Tensor.
    :return torch.Tensor: Latent embedding tensor of shape (B, T, d_e).
    :raises TypeError: If neither `output.embeddings` nor `output` itself is a torch.Tensor.
    """
    embeddings = getattr(output, "embeddings", output)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("Feature Module output is neither tensor nor .embeddings.")
    return embeddings


# =====================================================================
# EXPERIMENT ENTRY POINT
# =====================================================================

def reset_seeds() -> None:
    """
    Reset NumPy and PyTorch CPU/CUDA random seeds to the configured experiment SEED.

    :return None: Seeds random number generators in place.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def run_experiment() -> None:
    """
    Execute the two-model EINN data preparation, training, checkpoint export, and reporting pipeline.

    Validates model pair and observation window settings, generates and caches preprocessed
    training and held-out datasets, trains across configured conditioning/normalization modes,
    persists checkpoints and metrics, runs FeatureOnlyEvaluator reports, and writes a CSV summary.

    :return None: Saves datasets, model weights, metrics, figures, and run_summary.csv to disk.
    :raises ValueError: If OBSERVATION_WINDOWS, EMBEDDING_DIM, PCA_COMPONENTS_TO_SHOW, or
        MODEL_NAMES are invalid.
    """
    logging.basicConfig(level=logging.INFO)
    reset_seeds()

    if (
        len(OBSERVATION_WINDOWS) != 4
        or list(OBSERVATION_WINDOWS) != sorted(set(OBSERVATION_WINDOWS))
        or not all(1 <= n < SEQ_LEN for n in OBSERVATION_WINDOWS)
    ):
        raise ValueError("Exactly four increasing observation windows, each < SEQ_LEN, are needed.")
    if EMBEDDING_DIM < 2 or PCA_COMPONENTS_TO_SHOW < 1:
        raise ValueError("Use EMBEDDING_DIM >= 2 and PCA_COMPONENTS_TO_SHOW >= 1.")

    names = tuple(name.upper() for name in MODEL_NAMES)
    if set(names) not in ({"SIR", "SIS"}, {"SIR", "SEIR"}):
        raise ValueError("Choose MODEL_NAMES = ('SIR','SIS') or ('SIR','SEIR').")

    # Configure unified compartment/parameter dimensions and initial states for the selected pair.
    is_sir_sis = set(names) == {"SIR", "SIS"}
    d_s = 3 if is_sir_sis else 4
    d_p = 2 if is_sir_sis else 3
    initial_states = [0.99, 0.01, 0.0] if is_sir_sis else [0.99, 0.0, 0.01, 0.0]

    train_config = EINNTrainConfig(
        device=DEVICE,
        learning_rate=1e-3,
        supcon_temperature=0.1,
        mask_mismatched_losses=False,
        reps={"phase_1": 50, "phase_2": 50, "phase_3": 50, "phase_4": 50},
        loss_weights=LOSS_WEIGHTS.copy(),
    )

    # 1. Prepare training and held-out datasets using scaling statistics fitted on training data.
    data_loader = MultiModelInputLoader(device=DEVICE, seed=SEED)
    train_dataset, scaling_stats = data_loader.prepare_training_data(
        num_states_per_model=TRAIN_STATES_PER_MODEL,
        seq_len=SEQ_LEN,
        initial_states=initial_states,
        model_names=names,
        model_param_ranges={name: MODEL_PARAM_RANGES[name] for name in names},
        noise_std=NOISE_STD,
        window_min_days=20,
        window_step=20,
        pair_with_all_assumptions=False,
        fit_scalers=True,
    )
    test_dataset, _ = data_loader.prepare_training_data(
        num_states_per_model=TEST_STATES_PER_MODEL,
        seq_len=SEQ_LEN,
        initial_states=initial_states,
        model_names=names,
        model_param_ranges={name: MODEL_PARAM_RANGES[name] for name in names},
        noise_std=NOISE_STD,
        window_min_days=20,
        window_step=20,
        pair_with_all_assumptions=False,
        fit_scalers=False,
        scaling_stats=scaling_stats,
    )

    # Persist unexpanded trajectories, labels, and fitted scalers for offline reproducibility audits.
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    stored_trajectories: dict[str, Any] = {}
    for subset_name, dataset in (
        ("training", train_dataset),
        ("held_out", test_dataset),
    ):
        for source_name, values in (
            ("inputs", dataset.x_list),
            ("targets", dataset.y_list),
            ("times", dataset.t_list),
            ("auxiliary", dataset.aux_list),
        ):
            stored_trajectories[f"{subset_name}_{source_name}"] = np.stack(values)
        stored_trajectories[f"{subset_name}_true_labels"] = np.asarray(
            dataset.true_model_labels, dtype=np.int64
        )
        stored_trajectories[f"{subset_name}_assumed_labels"] = np.asarray(
            dataset.assumed_model_labels, dtype=np.int64
        )

    for scaler_name in ("x_scaler", "y_scaler", "aux_scaler"):
        scaler = getattr(data_loader.preprocessor, scaler_name)
        stored_trajectories[f"{scaler_name}_mean"] = scaler.mean_
        stored_trajectories[f"{scaler_name}_scale"] = scaler.scale_
    stored_trajectories["model_names"] = np.asarray(names)
    stored_trajectories["embedding_dimension"] = np.asarray(EMBEDDING_DIM)
    np.savez_compressed(
        os.path.join(OUTPUT_ROOT, "training_and_held_out_data.npz"),
        **stored_trajectories,
    )

    # 2. Train, save artifacts, and generate evaluation reports for each configured variant.
    results: list[dict[str, Any]] = []
    for condition_feature_on_true_label in FEATURE_LABEL_CONDITIONING_MODES:
        for normalize_feature in NORMALIZATION_MODES:
            reset_seeds()
            variant = "l2_normalized" if normalize_feature else "raw"
            conditioning = "label_conditioned" if condition_feature_on_true_label else "no_feature_label"
            output_dir = os.path.join(OUTPUT_ROOT, conditioning, variant, f"de_{EMBEDDING_DIM}")
            os.makedirs(output_dir, exist_ok=True)

            model_config = EINNModelConfig(
                d_x=5,
                d_e=EMBEDDING_DIM,
                d_s=d_s,
                d_p=d_p,
                feature_rnn_out=40,
                feature_bidirectional=True,
                num_models=2,
                model_label_dim=5,
                normalize_feature_embeddings=normalize_feature,
                condition_feature_on_true_label=condition_feature_on_true_label,
            )
            models = EINNBuilder.build_multi_einn(
                model_config=model_config,
                train_config=train_config,
                param_calibration={name: PARAM_CALIBRATION[name] for name in names},
                model_names=names,
                seed=SEED,
            )
            trainer = EINNTrainer(models, train_config, scaling_stats)
            metrics = trainer.train(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)

            post = TrainingPostProcessor(output_dir)
            post.save_models(models)
            post.save_scaling_stats(scaling_stats)
            post.save_metrics(metrics)

            evaluator = FeatureOnlyEvaluator(
                models=models,
                dataset=test_dataset,
                metrics=metrics,
                scaling_stats=scaling_stats,
                device=DEVICE,
                output_dir=output_dir,
                target_scaler=data_loader.preprocessor.y_scaler,
            )
            evaluator.generate_all_reports()
            results.append({
                "normalize_feature_embeddings": normalize_feature,
                "condition_feature_on_true_label": condition_feature_on_true_label,
                "embedding_dim": EMBEDDING_DIM,
                "output_dir": output_dir,
            })

    # 3. Write summary table of all executed experiment configurations.
    with open(os.path.join(OUTPUT_ROOT, "run_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
