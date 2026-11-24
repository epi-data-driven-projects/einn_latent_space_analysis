"""
Single-model EINN generalization experiment across observation windows and latent dimensions.

Trains and evaluates single-model EINNs (SIR, SIS, SEIRM, ROST) across four observed
prefix lengths (10, 18, 40, 80) on held-out synthetic trajectories, computes forecast
MSE, paired cosine similarity, Procrustes disparity/residuals, and shared/per-window PCA
statistics, renders the 4x5 generalization figure, and exports `.npz`, `.csv`, and
`.json` artifacts with fault-tolerant status tracking.

Run from the project root with: python einn_single_model_generalization_4x5.py
"""
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import matplotlib
import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.einn_input_loader import EINNInputLoader
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.inference.inference_engine import InferenceEngine
from einn.model.einn_builder import EINNBuilder
from einn.ode.rost_model import RostHungaryModel
from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel
from einn.ode.sis_model import SISModel
from einn.utils.experiment_utils import ExperimentUtils
from einn.visualization.single_generalization_plotter import SingleGeneralizationPlotter
from einn_runner import EINNRunner

matplotlib.use("Agg")  # Headless, repeatable saving during batch runs.

# ======================================================================
# EXPERIMENT CONFIGURATION
# ======================================================================

# Run all documented models and both latent dimensions when this switch is enabled.
RUN_ALL_MODELS: bool = True
ALL_MODEL_NAMES: tuple[str, ...] = ("SIR", "SIS", "ROST")
ALL_EMBEDDING_DIMS: tuple[int, ...] = (20, 5)

# Select a single model and embedding dimension when not running the full grid.
MODEL_NAME: str = "SIS"                 # "SIR", "SIS", "SEIRM", "ROST"
EMBEDDING_DIM: int = 20                 # Changing the latent dimension creates a distinct architecture and requires retraining.

# Display only the first configured principal components in diagnostic figures.
# The explained-variance thresholds are calculated using the full PCA basis.
PCA_COMPONENTS_TO_SHOW: int = 8

NUM_STATES: int = 100
SEQ_LEN: int = 200
OBSERVATION_WINDOWS: tuple[int, ...] = (10, 18, 40, 80)  # Evaluate the same four observed-prefix lengths for each held-out trajectory.
EPOCHS: int = 5
BATCH_SIZE: int = 32
SEED: int = 42
DEVICE: str = "cuda:1"                  # Change to "cuda:0" or "cpu".
BASE_OUTPUT_DIR: str = "./experiment_results/generalization"

NOISE_STD: dict[str, float] = {
    "SIR": 0.01,
    "SIS": 0.01,
    "SEIRM": 0.01,
    "ROST": 0.0005,
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "SIR": {
        "class": SIRModel,
        "target_idx": 1,
        "d_s": 3,
        "d_p": 2,
        "model_type": "SIR",
        "param_ranges": {
            "beta": (0.0, 1.0),
            "gamma": (0.0, 1.0),
        },
        "calibration": {"beta": 0.4, "gamma": 0.1},
        "test_params": [0.45, 0.1],
        "initial_states": [0.99, 0.01, 0.0],
        "target_name": "Infected",
    },
    "SIS": {
        "class": SISModel,
        "target_idx": 1,
        "d_s": 2,
        "d_p": 2,
        "model_type": "SIS",
        "param_ranges": {
            "beta": (0.0, 1.0),
            "gamma": (0.0, 1.0),
        },
        "calibration": {"beta": 0.4, "gamma": 0.1},
        "test_params": [0.45, 0.1],
        "initial_states": [0.99, 0.01],
        "target_name": "Infected",
    },
    "SEIRM": {
        "class": SEIRMModel,
        "target_idx": 2,
        "d_s": 5,
        "d_p": 4,
        "model_type": "SEIRM",
        "param_ranges": {
            "beta": (0.0, 1.0),
            "alpha": (0.0, 1.0),
            "gamma": (0.0, 1.0),
            "mu": (0.0, 1.0),
        },
        "calibration": {
            "beta": 0.4, "alpha": 0.2,
            "gamma": 0.1, "mu": 0.05,
        },
        "test_params": [0.45, 0.2, 0.1, 0.02],
        "initial_states": [0.99, 0.0, 0.01, 0.0, 0.0],
        "target_name": "Infected",
    },
    "ROST": {
        "class": RostHungaryModel,
        "target_idx": 15,
        "d_s": 18,
        "d_p": 14,
        "model_type": "ROST",
        "param_ranges": {
            "beta": (0.01, 1.0),
            "inf_a": (0.01, 1.0),
            "susc": (0.01, 1.0),
            "alpha_l": (0.01, 1.0),
            "alpha_p": (0.01, 1.0),
            "p": (0.01, 1.0),
            "gamma_a": (0.01, 1.0),
            "gamma_s": (0.01, 1.0),
            "h": (0.01, 1.0),
            "xi": (0.01, 1.0),
            "gamma_h": (0.01, 1.0),
            "gamma_c": (0.01, 1.0),
            "mu": (0.01, 1.0),
            "gamma_cr": (0.01, 1.0),
        },
        "calibration": {
            "beta": 0.4,
            "inf_a": 0.5,
            "susc": 0.9,
            "alpha_l": 0.2,
            "alpha_p": 0.3,
            "p": 0.4,
            "gamma_a": 0.1,
            "gamma_s": 0.1,
            "h": 0.05,
            "xi": 0.1,
            "gamma_h": 0.1,
            "gamma_c": 0.05,
            "mu": 0.02,
            "gamma_cr": 0.1,
        },
        "test_params": [
            0.45, 0.4, 0.95, 0.25, 0.25, 0.5, 0.15,
            0.15, 0.1, 0.08, 0.12, 0.05, 0.03, 0.1,
        ],
        # Same initial state actually used by the current training data loader:
        # S=0.99 and presymptomatic Ip=0.01 (index 3).
        "initial_states": [0.99, 0.0, 0.0, 0.01] + [0.0] * 14,
        "target_name": "Cumulative Infected",
    },
}

LOSS_WEIGHTS: dict[str, float] = {
    "data_T": 10.0,
    "data_F": 10.0,
    "aux": 8.0,
    "ode_T": 8.0,
    "ode_F": 8.0,
    "ode_future_T": 0.1,
    "ode_future_F": 0.1,
    "mono": 6.0,
    "param": 0.1,
    "kd_target": 10.0,
    "kd_emb": 10.0,
    "ic": 20.0,
    "ic_F": 20.0,
}

TEST_CORRELATIONS: dict[str, float] = {
    "mobility": -0.75,
    "temperature": -0.1,
    "testing": 0.85,
    "stringency": 0.95,
}


def select_device(requested: str) -> str:
    """
    Validate a requested PyTorch device string and fail early on invalid CUDA indices.

    :param str requested: Requested device specification (e.g. 'cpu', 'cuda:0', or 'cuda:1').
    :return str: Validated device string unchanged.
    :raises RuntimeError: If CUDA is requested but unavailable or the device index is out of range.
    """
    device = torch.device(requested)
    if device.type == "cuda":
        available = torch.cuda.device_count()
        index = 0 if device.index is None else device.index
        if not torch.cuda.is_available() or index >= available:
            raise RuntimeError(
                f"Requested {requested}, but {available} CUDA device(s) are available. "
                "Change DEVICE to an available GPU index or 'cpu'."
            )
    return requested


def make_data_cache_dir(model_name: str, config: dict[str, Any], output_root: Path) -> Path:
    """
    Create and return a deterministic data-cache directory keyed by data-generation settings.

    Excludes embedding_dim so runs at d_e = 5 and d_e = 20 for the same model and dataset
    configuration reuse the identical cached synthetic training trajectories.

    :param str model_name: Epidemiological model identifier (e.g. 'SIR', 'SIS', 'SEIRM', or 'ROST').
    :param dict[str, Any] config: Model configuration entry from MODEL_CONFIGS.
    :param Path output_root: Model-specific output directory under BASE_OUTPUT_DIR.
    :return Path: Path to the created cache directory `<output_root>/data_cache/<fingerprint>`.
    """
    cache_settings = {
        "model": model_name,
        "num_states": NUM_STATES,
        "seq_len": SEQ_LEN,
        "target_idx": config["target_idx"],
        "initial_states": config["initial_states"],
        "param_ranges": config["param_ranges"],
        "noise_std": NOISE_STD[model_name],
        "seed": SEED,
        "data_cache_version": 1,
    }
    fingerprint = hashlib.sha256(
        json.dumps(cache_settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = output_root / "data_cache" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_configuration.json").write_text(
        json.dumps(cache_settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cache_dir


def create_model(config: dict[str, Any]) -> Any:
    """
    Instantiate a mechanistic ODE model configured with `config['target_idx']`.

    :param dict[str, Any] config: Model configuration entry containing `'class'` and `'target_idx'`.
    :return Any: Instantiated ODE model object.
    """
    return config["class"](target_idx=config["target_idx"])


def load_training_feature_scaler(cache_dir: Path, config: dict[str, Any]) -> StandardScaler:
    """
    Load cached raw training covariates X and fit a StandardScaler strictly on training data.

    Prevents data leakage by fitting exclusively on `cached['x_train'][:NUM_STATES]` rather
    than held-out test trajectories.

    :param Path cache_dir: Cache directory containing `<NUM_STATES>_states_<SEQ_LEN>len_target_<idx>.pt`.
    :param dict[str, Any] config: Model configuration entry providing `'target_idx'`.
    :return StandardScaler: Fitted scikit-learn StandardScaler over the 5 input features.
    :raises FileNotFoundError: If the expected training cache file does not exist.
    :raises ValueError: If `x_train` has fewer than NUM_STATES trajectories or unexpected feature width.
    """
    filename = cache_dir / (
        f"{NUM_STATES}_states_{SEQ_LEN}len_target_{config['target_idx']}.pt"
    )
    if not filename.is_file():
        raise FileNotFoundError(
            f"Training cache missing: {filename}. "
            "Check EINNInputLoader's cache file naming convention."
        )
    # The file is generated by this experiment. Never load untrusted pickle files.
    cached = torch.load(filename, map_location="cpu", weights_only=False)
    x_train = cached["x_train"]
    if len(x_train) < NUM_STATES:
        raise ValueError("Training cache contains fewer states than requested.")
    stacked = np.concatenate(x_train[:NUM_STATES], axis=0)
    if stacked.ndim != 2 or stacked.shape[1] != 5:
        raise ValueError(f"Expected raw training X with 5 features, got {stacked.shape}.")
    return StandardScaler().fit(stacked)


def paired_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute timestep-wise cosine similarity between aligned embedding trajectories `a` and `b`.

    Produces NaN at timesteps where either vector has norm at or below 1e-12.

    :param np.ndarray a: First embedding array of shape (T, D).
    :param np.ndarray b: Second embedding array of shape (T, D) aligned along axis 0 with `a`.
    :return np.ndarray: 1D cosine similarity array of shape (T,).
    """
    numer = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numer, denom,
        out=np.full(a.shape[0], np.nan, dtype=float),
        where=denom > 1e-12,
    )


def extract_embeddings(output: object) -> torch.Tensor:
    """
    Extract a latent embedding tensor from a structured output object or raw torch.Tensor.

    :param object output: Module return value exposing `.embeddings` or a direct torch.Tensor.
    :return torch.Tensor: Extracted latent embedding tensor.
    :raises TypeError: If neither `output.embeddings` nor `output` is a torch.Tensor.
    """
    embeddings = getattr(output, "embeddings", output)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("Feature Module output is neither tensor nor .embeddings.")
    return embeddings


def evaluate_prefix(
    models: Any,
    x_test_scaled: np.ndarray,
    full_t_raw: torch.Tensor,
    scaling_stats: Any,
    true_target: np.ndarray,
    target_idx: int,
    t_end: int,
    device: str,
) -> dict[str, Any]:
    """
    Evaluate full-horizon forecasts, latent trajectories, cosine similarity, and Procrustes metrics
    for a single observed prefix length `t_end`.

    :param Any models: Built EINN module bundle containing time_module, feature_module,
        output_module, and ode_model.
    :param np.ndarray x_test_scaled: Standardized held-out covariate array of shape (SEQ_LEN, 5).
    :param torch.Tensor full_t_raw: Unscaled time grid tensor of shape (1, SEQ_LEN, 1).
    :param Any scaling_stats: Training-fitted scaling statistics for time scaling and unit conversion.
    :param np.ndarray true_target: Noise-free ground-truth target trajectory of shape (SEQ_LEN,).
    :param int target_idx: Target compartment index within the ODE state vector.
    :param int t_end: Number of initial observed timesteps passed to the Feature encoder.
    :param str device: Target PyTorch device string.
    :return dict[str, Any]: Per-window evaluation dictionary containing predictions, future MSE,
        Time/Feature latent arrays, cosine similarity, Procrustes disparity, and pointwise residuals.
    :raises ValueError: If `t_end` is out of range, shapes/finiteness checks fail, or predictions are invalid.
    :raises TypeError: If `InferenceEngine.predict_future` returns a list or tuple instead of a tensor.
    """
    if not 1 <= t_end < len(true_target):
        raise ValueError(f"Invalid observed prefix length {t_end}.")

    models.time_module.eval()
    models.feature_module.eval()
    models.output_module.eval()
    models.ode_model.eval()

    x_prefix = torch.as_tensor(
        x_test_scaled[:t_end], dtype=torch.float32, device=device
    ).unsqueeze(0)
    time_raw = full_t_raw.to(device)

    with torch.no_grad():
        states = InferenceEngine(models).predict_future(
            historical_x=x_prefix,
            historical_x_mask=None,
            future_t=time_raw,
            scaling_stats=scaling_stats,
        )
        if isinstance(states, (tuple, list)):
            raise TypeError("InferenceEngine returned an unexpected compound object.")
        pred = states[0, :, target_idx].detach().cpu().numpy()

        time_min = scaling_stats.t_min.to(device)
        time_max = scaling_stats.t_max.to(device)
        scaled_t = (time_raw - time_min) / (time_max - time_min).clamp_min(1e-8)

        e_time = models.time_module(t=scaled_t)[0].detach().cpu().numpy()
        output_feature = models.feature_module(
            x=x_prefix, t=scaled_t, mask=None
        )
        e_feature = extract_embeddings(output_feature)[0].detach().cpu().numpy()

    if e_time.shape != e_feature.shape:
        raise ValueError(f"Embedding shape mismatch: {e_time.shape} vs {e_feature.shape}.")
    if e_time.ndim != 2 or e_time.shape[0] != len(true_target):
        raise ValueError(f"Unexpected embedding matrix: {e_time.shape}.")
    if not (np.isfinite(e_time).all() and np.isfinite(e_feature).all()):
        raise ValueError("Non-finite embeddings; investigate network training.")
    if pred.shape != true_target.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid physical-scale prediction or target shape.")

    # Compute MSE strictly over unseen future steps [t_end:] in physical units.
    mse_future = float(np.mean((pred[t_end:] - true_target[t_end:]) ** 2))
    cosine = paired_cosine_similarity(e_time, e_feature)

    # Guard against degenerate zero-variance point clouds that cause scipy.spatial.procrustes to fail.
    if (
        np.linalg.norm(e_time - e_time.mean(axis=0)) <= 1e-12
        or np.linalg.norm(e_feature - e_feature.mean(axis=0)) <= 1e-12
    ):
        procrustes_disparity = float("nan")
        procrustes_residual = np.full(len(true_target), np.nan)
    else:
        aligned_time, aligned_feature, procrustes_disparity = procrustes(
            e_time, e_feature
        )
        procrustes_residual = np.linalg.norm(
            aligned_time - aligned_feature, axis=1
        )

    return {
        "t_end": t_end,
        "predicted_trajectory": pred,
        "mse_future": mse_future,
        "latent_time": e_time,
        "latent_feature": e_feature,
        "cosine_similarity": cosine,
        "procrustes_disparity": float(procrustes_disparity),
        "procrustes_pointwise_residual": procrustes_residual,
    }


def fit_shared_pca(phase_results: list[dict[str, Any]]) -> PCA:
    """
    Fit a shared 2D PCA projection across all observation windows and per-window full PCA curves.

    Populates each entry in `phase_results` with `'pca_coords'` (in the joint 2D basis),
    `'cum_variance'`, `'pca_95'`, and `'pca_99'` (from a window-local full PCA fit).

    :param list[dict[str, Any]] phase_results: List of per-window result dicts containing `'latent_feature'`.
    :return PCA: Two-component PCA estimator fitted on pooled Feature embeddings across all windows.
    """
    all_features = np.concatenate(
        [result["latent_feature"] for result in phase_results], axis=0
    )
    shared = PCA(n_components=2)
    shared.fit(all_features)

    for result in phase_results:
        features = result["latent_feature"]
        result["pca_coords"] = shared.transform(features)

        phase_pca = PCA(n_components=min(features.shape))
        phase_pca.fit(features)
        cumulative = np.cumsum(phase_pca.explained_variance_ratio_)
        result["cum_variance"] = cumulative
        result["pca_95"] = int(np.searchsorted(cumulative, 0.95) + 1)
        result["pca_99"] = int(np.searchsorted(cumulative, 0.99) + 1)
    return shared


def json_write(path: Path, data: dict[str, Any]) -> None:
    """
    Write a dictionary as formatted UTF-8 JSON to `path`.

    :param Path path: Destination file path.
    :param dict[str, Any] data: JSON-serializable dictionary (non-JSON types converted via `str`).
    :return None: Writes the JSON file to disk.
    """
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_completed_experiment(
    output_dir: Path,
    phase_results: list[dict[str, Any]],
    shared_pca: PCA,
    true_target: np.ndarray,
    data_test: dict[str, Any],
    x_test_scaled: np.ndarray,
    scaler: StandardScaler,
) -> None:
    """
    Persist window metrics, compressed `.npz` embedding/prediction archives, test inputs, and PCA curves.

    Writes `metrics_by_window.csv`, `test_latent_representations.npz`,
    `test_data_and_scaler.npz`, and `pca_cumulative_variance_by_window.csv` into `output_dir`.

    :param Path output_dir: Target directory for serialized experiment artifacts.
    :param list[dict[str, Any]] phase_results: Per-window evaluation results with PCA fields attached.
    :param PCA shared_pca: Shared 2D PCA estimator fitted across all windows.
    :param np.ndarray true_target: Ground-truth noise-free target series of shape (SEQ_LEN,).
    :param dict[str, Any] data_test: Synthetic test output dictionary containing `'x_raw'` and `'true_latent_states'`.
    :param np.ndarray x_test_scaled: Standardized test covariate array of shape (SEQ_LEN, 5).
    :param StandardScaler scaler: Training-fitted StandardScaler used on input features.
    :return None: Writes CSV and NPZ files to `output_dir`.
    """
    import csv

    metrics_file = output_dir / "metrics_by_window.csv"
    with metrics_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow((
            "observed_steps", "future_mse_physical", "pca_d95", "pca_d99",
            "procrustes_disparity", "mean_cosine", "pca_2d_pc1_ratio",
            "pca_2d_pc2_ratio",
        ))
        for r in phase_results:
            writer.writerow((
                r["t_end"], r["mse_future"], r["pca_95"], r["pca_99"],
                r["procrustes_disparity"], float(np.nanmean(r["cosine_similarity"])),
                float(shared_pca.explained_variance_ratio_[0]),
                float(shared_pca.explained_variance_ratio_[1]),
            ))

    np.savez_compressed(
        output_dir / "test_latent_representations.npz",
        observation_windows=np.asarray(OBSERVATION_WINDOWS),
        true_target=true_target,
        time_embeddings=np.stack([r["latent_time"] for r in phase_results]),
        feature_embeddings=np.stack([r["latent_feature"] for r in phase_results]),
        predictions=np.stack([r["predicted_trajectory"] for r in phase_results]),
        cosine_similarities=np.stack([r["cosine_similarity"] for r in phase_results]),
        shared_pca_coords=np.stack([r["pca_coords"] for r in phase_results]),
        shared_pca_components=shared_pca.components_,
        shared_pca_mean=shared_pca.mean_,
        shared_pca_explained_variance_ratio=shared_pca.explained_variance_ratio_,
        cumulative_pca_variance=np.stack([r["cum_variance"] for r in phase_results]),
        procrustes_residuals=np.stack([r["procrustes_pointwise_residual"] for r in phase_results]),
    )
    np.savez_compressed(
        output_dir / "test_data_and_scaler.npz",
        x_test_raw=np.asarray(data_test["x_raw"]),
        x_test_scaled=x_test_scaled,
        test_true_latent_states=np.asarray(data_test["true_latent_states"]),
        training_x_scaler_mean=scaler.mean_,
        training_x_scaler_scale=scaler.scale_,
        training_x_scaler_var=scaler.var_,
    )
    # Write cumulative explained variance separately for each observed prefix.
    with (output_dir / "pca_cumulative_variance_by_window.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["component"] + [f"obs_{n}" for n in OBSERVATION_WINDOWS])
        curves = np.stack([r["cum_variance"] for r in phase_results])
        for i in range(curves.shape[1]):
            writer.writerow([i + 1] + [float(v) for v in curves[:, i]])


def run_generalization_experiment(model_name: str, embedding_dim: int) -> dict[str, Any]:
    """
    Run training, held-out prefix evaluation, PCA analysis, plotting, and artifact persistence
    for a single epidemiological model and latent embedding dimension.

    :param str model_name: Model key in MODEL_CONFIGS ('SIR', 'SIS', 'SEIRM', or 'ROST').
    :param int embedding_dim: Latent embedding dimension d_e (5 or 20).
    :return dict[str, Any]: Summary dictionary containing status, output_dir, mean_future_mse,
        and mean_pca_d95.
    :raises ValueError: If `model_name`, `embedding_dim`, `PCA_COMPONENTS_TO_SHOW`, or
        `OBSERVATION_WINDOWS` is invalid.
    :raises FileNotFoundError: If a saved model checkpoint is missing after training.
    :raises Exception: Re-raises any error after recording status `'hiba'` and writing `hiba_reszletei.txt`.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Ismeretlen epidemiológiai modell: {model_name!r}.")
    if embedding_dim not in (5, 20):
        raise ValueError("A beágyazás dimenziója csak 5 vagy 20 lehet.")
    if not 1 <= PCA_COMPONENTS_TO_SHOW <= 20:
        raise ValueError("PCA_COMPONENTS_TO_SHOW értéke 1 és 20 között lehet.")
    if (
        len(OBSERVATION_WINDOWS) != 4
        or sorted(set(OBSERVATION_WINDOWS)) != list(OBSERVATION_WINDOWS)
        or not all(1 <= n < SEQ_LEN for n in OBSERVATION_WINDOWS)
    ):
        raise ValueError("Négy növekvő megfigyelési ablak szükséges, mind < SEQ_LEN.")

    device = select_device(DEVICE)
    ExperimentUtils.seed_everything(seed=SEED)
    config = MODEL_CONFIGS[model_name]
    output_root = Path(BASE_OUTPUT_DIR) / model_name.lower()
    output_dir = output_root / f"de_{embedding_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_cache_dir = make_data_cache_dir(model_name, config, output_root)
    target_idx = config["target_idx"]

    run_settings: dict[str, Any] = {
        "model": model_name,
        "embedding_dim": embedding_dim,
        "num_training_trajectories": NUM_STATES,
        "sequence_length": SEQ_LEN,
        "observation_windows": OBSERVATION_WINDOWS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "device": device,
        "target_idx": target_idx,
        "training_parameter_ranges": config["param_ranges"],
        "training_calibration": config["calibration"],
        "test_parameters": config["test_params"],
        "test_initial_states": config["initial_states"],
        "noise_std": NOISE_STD[model_name],
        "loss_weights": LOSS_WEIGHTS,
        "pca_components_plotted": PCA_COMPONENTS_TO_SHOW,
        "training_data_cache": str(data_cache_dir.resolve()),
        "status": "folyamatban",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    json_write(output_dir / "experiment_config.json", run_settings)

    print("\n" + "═" * 76)
    print(f"MODELL: {model_name}  |  LÁTENS DIMENZIÓ: {embedding_dim}")
    print(f"Tanítóadat-gyorsítótár: {data_cache_dir}")
    print(f"Eredmények: {output_dir.resolve()}")
    print("═" * 76)

    try:
        loader = EINNInputLoader(
            true_model=create_model(config),
            calibrator_model=create_model(config),
            target_idx=target_idx,
            device=device,
            cache_dir=str(data_cache_dir),
        )
        dataset, scaling_stats = loader.prepare_training_data(
            num_states=NUM_STATES,
            seq_len=SEQ_LEN,
            d_p=config["d_p"],
            param_ranges=config["param_ranges"],
            initial_states=config["initial_states"],
            noise_std=NOISE_STD[model_name],
        )

        model_cfg = EINNModelConfig(
            d_x=5,
            d_e=embedding_dim,
            d_s=config["d_s"],
            d_p=config["d_p"],
            feature_n_layers=2,
        )
        train_cfg = EINNTrainConfig(
            device=device,
            learning_rate=1e-3,
            early_stopping_patience=5,
            loss_weights=LOSS_WEIGHTS.copy(),
            reps={"phase_1": 100, "phase_2": 100, "phase_3": 100, "phase_4": 100},
        )
        runner = EINNRunner(
            output_dir=str(output_dir),
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            seed=SEED,
        )
        runner.train_and_save(
            dataset=dataset,
            scaling_stats=scaling_stats,
            model_type=config["model_type"],
            target_idx=target_idx,
            param_calibration=config["calibration"],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
        # Record successful training before evaluating the test trajectory.
        run_settings["status"] = "betanítva"
        json_write(output_dir / "experiment_config.json", run_settings)

        # Generate a held-out trajectory with a freshly sampled observation-noise realization.
        generator_test = SyntheticDataGenerator(
            true_ode_model=create_model(config),
            seq_len=SEQ_LEN,
            rng=np.random.default_rng(SEED + 1001),
        )
        data_test = generator_test.generate(
            initial_states=config["initial_states"],
            true_params=config["test_params"],
            noise_std=NOISE_STD[model_name],
            correlations=TEST_CORRELATIONS,
        )
        true_target = np.asarray(
            data_test["true_latent_states"][:, target_idx], dtype=float
        ).reshape(-1)
        scaler = load_training_feature_scaler(data_cache_dir, config)
        x_test_scaled = scaler.transform(data_test["x_raw"])

        # Reload the saved weights to evaluate exactly the exported network state.
        models = EINNBuilder.build_einn(
            model_config=model_cfg,
            train_config=train_cfg,
            param_calibration=config["calibration"],
            target_idx=target_idx,
            model_type=config["model_type"],
            seed=SEED,
        )
        for attr, file_name in (
            ("time_module", "einn_time_module.pth"),
            ("feature_module", "einn_feature_module.pth"),
            ("output_module", "einn_output_module.pth"),
        ):
            checkpoint = output_dir / file_name
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Hiányzó modellcheckpoint: {checkpoint}. "
                    "Ellenőrizd az EINNRunner által mentett fájlneveket."
                )
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            getattr(models, attr).load_state_dict(state, strict=True)

        full_t = torch.arange(
            SEQ_LEN, dtype=torch.float32, device=device
        ).view(1, SEQ_LEN, 1)
        phase_results: list[dict[str, Any]] = []
        for end in OBSERVATION_WINDOWS:
            print(f"Kiértékelés: {end} megfigyelt időpont")
            result = evaluate_prefix(
                models=models,
                x_test_scaled=x_test_scaled,
                full_t_raw=full_t,
                scaling_stats=scaling_stats,
                true_target=true_target,
                target_idx=target_idx,
                t_end=end,
                device=device,
            )
            phase_results.append(result)
            # Persist the window-level predictions and embeddings before completing visualization.
            np.savez_compressed(
                output_dir / f"ablak_{end}_ideiglenes_eredmenyek.npz",
                observed_steps=end,
                true_target=true_target,
                prediction=result["predicted_trajectory"],
                time_embedding=result["latent_time"],
                feature_embedding=result["latent_feature"],
                cosine_similarity=result["cosine_similarity"],
                procrustes_residual=result["procrustes_pointwise_residual"],
            )

        shared_pca = fit_shared_pca(phase_results)
        plot_path = output_dir / (
            f"{model_name.lower()}_altalanositas_de_{embedding_dim}_4x5.png"
        )
        plot_results(
            phase_results=phase_results,
            shared_pca=shared_pca,
            true_target=true_target,
            target_name=(
                "Kumulált fertőzöttek aránya"
                if model_name == "ROST" else "Fertőzöttek aránya"
            ),
            output_path=plot_path,
            model_name=model_name,
            embedding_dim=embedding_dim,
        )
        save_completed_experiment(
            output_dir=output_dir,
            phase_results=phase_results,
            shared_pca=shared_pca,
            true_target=true_target,
            data_test=data_test,
            x_test_scaled=x_test_scaled,
            scaler=scaler,
        )
        run_settings["status"] = "sikeres"
        run_settings["finished_at"] = datetime.now().isoformat(timespec="seconds")
        run_settings["figure_png"] = plot_path.name
        run_settings["figure_pdf"] = plot_path.with_suffix(".pdf").name
        json_write(output_dir / "experiment_config.json", run_settings)

        results_for_manifest: dict[str, Any] = {
            "model": model_name,
            "embedding_dim": embedding_dim,
            "status": "sikeres",
            "output_dir": str(output_dir.resolve()),
            "mean_future_mse": float(np.mean([r["mse_future"] for r in phase_results])),
            "mean_pca_d95": float(np.mean([r["pca_95"] for r in phase_results])),
        }
        print(f"\nÁbra mentve (PNG): {plot_path.resolve()}")
        print(f"Ábra mentve (PDF): {plot_path.with_suffix('.pdf').resolve()}")
        print(f"Mérőszámok: {(output_dir / 'metrics_by_window.csv').resolve()}")
        for r in phase_results:
            print(
                f"{r['t_end']:>3} megfigyelés | "
                f"jövőbeli négyzetes hiba={r['mse_future']:.6g} | "
                f"d95={r['pca_95']} | d99={r['pca_99']} | "
                f"Procrustes={r['procrustes_disparity']:.5g}"
            )
        return results_for_manifest
    except Exception as exc:
        run_settings["status"] = "hiba"
        run_settings["finished_at"] = datetime.now().isoformat(timespec="seconds")
        run_settings["error"] = f"{type(exc).__name__}: {exc}"
        json_write(output_dir / "experiment_config.json", run_settings)
        (output_dir / "hiba_reszletei.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        # Release cached Python and CUDA memory between independent experiment runs.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    """
    Execute all planned single-model generalization runs and write `all_runs_summary.json`.

    :return None: Runs selected `(model_name, embedding_dim)` pairs and saves summary metadata.
    :raises RuntimeError: If one or more planned runs fail.
    """
    plans = (
        [(model_name, d) for model_name in ALL_MODEL_NAMES for d in ALL_EMBEDDING_DIMS]
        if RUN_ALL_MODELS else [(MODEL_NAME, EMBEDDING_DIM)]
    )
    root = Path(BASE_OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []
    print(f"Tervezett futtatások száma: {len(plans)}")
    for run_index, (model_name, embedding_dim) in enumerate(plans, 1):
        print(f"\n[{run_index}/{len(plans)}] {model_name}, d_e={embedding_dim}")
        try:
            result = run_generalization_experiment(model_name, embedding_dim)
        except Exception as exc:
            result = {
                "model": model_name,
                "embedding_dim": embedding_dim,
                "status": "hiba",
                "output_dir": str((root / model_name.lower() / f"de_{embedding_dim}").resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"HIBA a(z) {model_name}, d_e={embedding_dim} futtatásnál: {exc}")
            if not RUN_ALL_MODELS:
                report.append(result)
                json_write(root / "all_runs_summary.json", {"runs": report})
                raise
        report.append(result)
        json_write(root / "all_runs_summary.json", {"runs": report})

    successes = sum(r["status"] == "sikeres" for r in report)
    print(f"\nKész: {successes}/{len(report)} futtatás sikeres.")
    if successes != len(report):
        raise RuntimeError(
            f"{len(report) - successes} futtatás hibával ért véget. "
            f"Részletek: {root / 'all_runs_summary.json'}"
        )


# The figure remains generated by the original plotting method.
plot_results = SingleGeneralizationPlotter().plot_results

if __name__ == "__main__":
    main()
