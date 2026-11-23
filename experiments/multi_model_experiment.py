
"""True-model-only two-model EINN experiment with single-model-style 4x4 reports.

What this adds compared to the original script:
- configurable EMBEDDING_DIM, OBSERVATION_WINDOWS, PCA_COMPONENTS_TO_SHOW
- two extra figures after training/evaluation:
  one 4x4 figure for a held-out trajectory of model_1,
  one 4x4 figure for a held-out trajectory of model_2
- each row = one observation window
- columns: prediction, PCA explained variance, shared PCA 2D, cosine similarity
- Procrustes values appear in a full-width bar chart below the 4x4 grid
- an additional figure shows ALL independent held-out trajectories in one common
  PCA space (model-colored curves), with original-dimensional Hausdorff metrics
- plotting style exactly follows the single-model 4x4 reference
- saved figures: PNG and vector PDF; numeric results: CSV and NPZ

Time Module routing: models.time_modules[true_model_name], as in the
included model builder. No classifier is used for unknown model labels.
"""

import csv
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import torch
from scipy.spatial import procrustes, cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from einn_multimodel.config.einn_model_config import EINNModelConfig
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.data.multi_model_input_loader import MultiModelInputLoader
from einn_multimodel.inference.training_post_processor import TrainingPostProcessor
from einn_multimodel.inference.inference_engine import InferenceEngine
from einn_multimodel.model.einn_builder import EINNBuilder
from einn_multimodel.training.einn_trainer import EINNTrainer


# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

SEED = 42
MODEL_NAMES = ("SIR", "SIS")   # or ("SIR", "SIS")
TRAIN_STATES_PER_MODEL = 100
TEST_STATES_PER_MODEL = 20
SEQ_LEN = 200
NOISE_STD = 0.01
EPOCHS = 5
BATCH_SIZE = 32
DEVICE = "cuda:1"

EMBEDDING_DIM = 20                 # 5 or 20
OBSERVATION_WINDOWS = (10, 18, 40, 80)
PCA_COMPONENTS_TO_SHOW = 8

NORMALIZATION_MODES = (False,)
FEATURE_LABEL_CONDITIONING_MODES = (True,)
OUTPUT_ROOT = f"./outputs/{'_'.join(n.lower() for n in MODEL_NAMES)}_generalization_4x4"

PARAM_CALIBRATION = {
    "SIR": {"beta": 0.4, "gamma": 0.1},
    "SIS": {"beta": 0.4, "gamma": 0.1},
    "SEIR": {"beta": 0.4, "alpha": 0.2, "gamma": 0.1},
}
MODEL_PARAM_RANGES = {
    "SIR": {"beta": (0.0, 1.0), "gamma": (0.0, 1.0)},
    "SIS": {"beta": (0.0, 1.0), "gamma": (0.0, 1.0)},
    "SEIR": {"beta": (0.0, 1.0), "alpha": (0.0, 1.0), "gamma": (0.0, 1.0)},
}

LOSS_WEIGHTS = {
    "data_T": 4.0,
    "data_F": 4.0,
    "aux": 4.0,
    "ode_T": 10.0,
    "ode_F": 10.0,
    "ode_future_T": 10.0,
    "ode_future_F": 10.0,
    "mono": 1.0,
    "param": 0.001,
    "kd_target": 8.0,
    "kd_emb": 8.0,
    "ic": 10.0,
    "ic_F": 10.0,
    "supcon": 5.0,
}


# =====================================================================
# VISUAL STYLE
# =====================================================================

PALETTE = {
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
PNG_DPI = 230


















# =====================================================================
# METRIC HELPERS
# =====================================================================

def paired_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the row-wise cosine similarity between aligned latent vectors.


    :param np.ndarray a: First operand or trajectory in the numerical comparison.
    :param np.ndarray b: Second operand or trajectory in the numerical comparison.
    :return np.ndarray: Computed result in the configured output representation.
    """
    numer = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numer, denom,
        out=np.full(a.shape[0], np.nan, dtype=float),
        where=denom > 1e-12,
    )


def hausdorff_statistics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute bidirectional nearest-neighbor distances and Hausdorff summaries.


    :param np.ndarray a: First operand or trajectory in the numerical comparison.
    :param np.ndarray b: Second operand or trajectory in the numerical comparison.
    :return dict: Named tensors, metrics, or model components produced by this operation.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    distance_a_to_b = cKDTree(b).query(a, k=1, workers=1)[0]
    distance_b_to_a = cKDTree(a).query(b, k=1, workers=1)[0]
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
    """Retrieve extract embeddings from the configured model or saved dataset.


    :param object output: Network predictions or output container used by this operation.
    :return torch.Tensor: Tensor containing the computed model outputs or time derivatives.
    """
    embeddings = getattr(output, "embeddings", output)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("Feature Module output is neither tensor nor .embeddings.")
    return embeddings


# =====================================================================
# EVALUATOR
# =====================================================================



# =====================================================================
# EXPERIMENT ENTRY POINT
# =====================================================================

def reset_seeds():
    """Reset all available numerical random generators to the experiment seed.


    :return Any: The computed objects, metrics, or artifact paths for the requested operation.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def run_experiment():
    """Train, evaluate, and save the selected epidemiological model experiment.


    :return Any: The computed objects, metrics, or artifact paths for the requested operation.
    """
    logging.basicConfig(level=logging.INFO)
    reset_seeds()
    if len(OBSERVATION_WINDOWS) != 4 or list(OBSERVATION_WINDOWS) != sorted(set(OBSERVATION_WINDOWS)) or not all(1 <= n < SEQ_LEN for n in OBSERVATION_WINDOWS):
        raise ValueError("Exactly four increasing observation windows, each < SEQ_LEN, are needed.")
    if not 2 <= EMBEDDING_DIM or PCA_COMPONENTS_TO_SHOW < 1:
        raise ValueError("Use EMBEDDING_DIM >= 2 and PCA_COMPONENTS_TO_SHOW >= 1.")
    names = tuple(name.upper() for name in MODEL_NAMES)
    if set(names) not in ({"SIR", "SIS"}, {"SIR", "SEIR"}):
        raise ValueError("Choose MODEL_NAMES = ('SIR','SIS') or ('SIR','SEIR').")
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

    # Save the complete preprocessed trajectories before window expansion so
    # held-out tests can be audited without generating new random realizations.
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    stored_trajectories = {}
    for subset_name, dataset in (
        ("training", train_dataset), ("held_out", test_dataset),
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

    results = []
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

    with open(os.path.join(OUTPUT_ROOT, "run_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

