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
TRAIN_STATES_PER_MODEL = 2
TEST_STATES_PER_MODEL = 1
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


def configure_plot_style() -> None:
    """Exact visual language of the single-model 4x4 reference script."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": "#AAB8C8",
        "axes.labelcolor": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.8,
        "grid.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _finite_limits(values, *, floor=None, ceiling=None,
                   relative_pad=.12, min_span=.015):
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-.05, 1.05)
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, min_span)
    pad = max(span * relative_pad, min_span * .30)
    lo, hi = lo - pad, hi + pad
    if floor is not None:
        lo = max(lo, floor)
    if ceiling is not None:
        hi = min(hi, ceiling)
    if hi <= lo:
        mid = float(v.mean())
        lo = max(floor, mid - .02) if floor is not None else mid - .02
        hi = min(ceiling, mid + .02) if ceiling is not None else mid + .02
    return lo, hi


def _panel_style(ax, time_axis=False):
    ax.set_axisbelow(True)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", labelsize=9, length=3)
    if time_axis:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def _mark_observed(ax, n, horizon):
    ax.axvspan(-.5, n - .5, color=PALETTE["background_obs"], alpha=.70, zorder=0)
    ax.axvline(n - .5, color=PALETTE["muted"], ls=(0, (4, 3)), lw=1.4, zorder=4)
    ax.set_xlim(0, horizon - 1)


def _pca_axis(ax, cum, d95, d99, max_components):
    show = min(max_components, cum.size)
    x = np.arange(1, show + 1)
    y = cum[:show]
    ax.plot(x, y, color=PALETTE["blue"], lw=3.6, marker="o", markersize=6.4,
            markerfacecolor="white", markeredgewidth=1.8, zorder=4)
    ax.axhline(.95, color=PALETTE["orange"], ls="--", lw=2.8, zorder=3)
    ax.axhline(.99, color=PALETTE["magenta"], ls=(0, (3, 2)), lw=2.8, zorder=3)
    ax.set_ylim(*_finite_limits(np.r_[y, .95, .99], floor=0., ceiling=1.005,
                                relative_pad=.06, min_span=.03))
    ax.set_xlim(.65, show + .4)
    ax.set_xticks(x)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    ax.set_title(f"PCA · 95%: {d95} és 99%: {d99} komponens", pad=9)
    ax.set_xlabel("Főkomponensek száma")
    ax.set_ylabel("Kumulatív variancia")
    _panel_style(ax)


def _cosine_axis(ax, values, times, n, horizon):
    ax.plot(times, values, color=PALETTE["navy"], lw=3.1,
            solid_joinstyle="round", zorder=3)
    _mark_observed(ax, n, horizon)
    limits = _finite_limits(values, floor=-1., ceiling=1.,
                            relative_pad=.09, min_span=.0005)
    ax.set_ylim(*limits)
    ax.set_title("Time–Feature · koszinuszos hasonlóság", pad=9)
    ax.set_xlabel("Időindex")
    ax.set_ylabel("Koszinuszos hasonlóság")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    _panel_style(ax, time_axis=True)
    digits = 5 if (limits[1] - limits[0]) < .03 else 3
    ax.text(.98, .03, f"{limits[0]:.{digits}f} – {limits[1]:.{digits}f}",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=8,
            color=PALETTE["muted"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=.75, pad=2))


def _procrustes_bars(ax, phase_results):
    obs = [int(r["t_end"]) for r in phase_results]
    vals = np.asarray([r["procrustes_disparity"] for r in phase_results], dtype=float)
    yp = np.arange(len(obs))
    colors = [PALETTE["teal_dark"], PALETTE["blue"],
              PALETTE["orange"], PALETTE["magenta"]]
    good = vals[np.isfinite(vals) & (vals >= 0)]
    xmax = max(float(good.max()) * 1.22 if good.size else 0., .01)
    ax.barh(yp, np.where(np.isfinite(vals), vals, 0), height=.61,
            color=colors[:len(obs)], alpha=.92)
    ax.set_yticks(yp, labels=[f"{n} megfigyelés" for n in obs])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Procrustes-eltérés (kisebb érték: jobb geometriai illeszkedés)")
    ax.set_title("A megfelelő Time és Feature beágyazások Procrustes-eltérése",
                 pad=12, fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, val in enumerate(vals):
        txt = f"{val:.6f}" if np.isfinite(val) else "nem számolható"
        ax.text((float(val) if np.isfinite(val) else 0.) + xmax*.011,
                i, txt, va="center", ha="left",
                fontsize=10.5, fontweight="bold", color=PALETTE["ink"])


def _save_plot(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=.18)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


# =====================================================================
# METRIC HELPERS
# =====================================================================

def paired_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numer = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numer, denom,
        out=np.full(a.shape[0], np.nan, dtype=float),
        where=denom > 1e-12,
    )


def hausdorff_statistics(a: np.ndarray, b: np.ndarray) -> dict:
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
    embeddings = getattr(output, "embeddings", output)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("Feature Module output is neither tensor nor .embeddings.")
    return embeddings


# =====================================================================
# EVALUATOR
# =====================================================================

class FeatureOnlyEvaluator:
    def __init__(self, models, dataset, metrics, scaling_stats, device, output_dir, target_scaler):
        self.models = models
        self.dataset = dataset
        self.metrics = metrics
        self.scaling_stats = scaling_stats
        self.target_scaler = target_scaler
        if target_scaler is None or not hasattr(target_scaler, "inverse_transform"):
            raise ValueError("Pass the training-fitted y_scaler.")
        self.device = device
        self.output_dir = output_dir
        self.id_to_model = {v: k for k, v in models.model_to_id.items()}
        self.class_ids = sorted(self.id_to_model)
        target_indices = {model.target_idx for model in models.ode_models.values()}
        if len(target_indices) != 1:
            raise ValueError("All models must agree on the target compartment index.")
        self.target_idx = next(iter(target_indices))
        if not np.allclose(
            [float(scaling_stats.aux_mean[self.target_idx]),
             float(scaling_stats.aux_std[self.target_idx])],
            [float(target_scaler.mean_[0]), float(target_scaler.scale_[0])],
            rtol=1e-3, atol=1e-6,
        ):
            raise ValueError("Incompatible train-fitted AUX and target scaling.")
        os.makedirs(output_dir, exist_ok=True)

    def scale_time(self, t):
        minimum = self.scaling_stats.t_min.to(self.device)
        maximum = self.scaling_stats.t_max.to(self.device)
        return (t - minimum) / (maximum - minimum).clamp_min(1e-8)

    def clean_target_physical(self, index: int) -> np.ndarray:
        scaled = np.asarray(self.dataset.y_list[index], dtype=np.float64).reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled).reshape(-1)

    def get_time_module_for_label(self, true_id: int):
        """Return the correct Time Module for the TRUE model label.

        If your local object names differ, most likely ONLY this function needs editing.
        Expected structure: models.time_modules is a dict keyed by model name.
        """
        model_name = self.id_to_model[int(true_id)]
        if hasattr(self.models, "time_modules"):
            return self.models.time_modules[model_name]
        if hasattr(self.models, "time_module_dict"):
            return self.models.time_module_dict[model_name]
        raise AttributeError(
            "Could not find time module dictionary on models. "
            "Adjust get_time_module_for_label()."
        )

    def collect_full_trajectories(self, all_trajectories: bool = False):
        required = ("x_list", "t_list", "true_model_labels")
        if any(not hasattr(self.dataset, field) for field in required):
            raise ValueError("Test dataset has no full-trajectory attributes.")
        if all_trajectories:
            selected = list(range(len(self.dataset.x_list)))
        else:
            selected = []
            for true_id in self.class_ids:
                matches = [i for i, label in enumerate(self.dataset.true_model_labels)
                           if int(label) == true_id]
                if not matches:
                    raise RuntimeError(f"Missing held-out trajectory for {true_id}.")
                selected.append(matches[0])

        embedding_parts, true_parts, trajectory_parts, time_parts = [], [], [], []
        self.models.feature_module.eval()
        with torch.no_grad():
            for trajectory_id, idx in enumerate(selected):
                x = torch.as_tensor(self.dataset.x_list[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                t = torch.as_tensor(self.dataset.t_list[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                true_id = int(self.dataset.true_model_labels[idx])
                labels = torch.tensor([true_id], dtype=torch.long, device=self.device)
                e_f = self.models.feature_module(
                    x=x, t=self.scale_time(t), mask=None, true_model_label=labels,
                ).embeddings.squeeze(0).cpu().numpy()
                n_steps = len(e_f)
                embedding_parts.append(e_f)
                true_parts.extend([true_id] * n_steps)
                trajectory_parts.extend([trajectory_id] * n_steps)
                time_parts.extend(range(n_steps))

        return {
            "feature_embeddings": np.concatenate(embedding_parts),
            "true_labels": np.asarray(true_parts, dtype=int),
            "trajectory_ids": np.asarray(trajectory_parts, dtype=int),
            "time_indices": np.asarray(time_parts, dtype=int),
            "selected_dataset_indices": np.asarray(selected, dtype=int),
        }

    def evaluate_hausdorff_test_embeddings(self):
        data = self.collect_full_trajectories(all_trajectories=True)
        E = data["feature_embeddings"]
        a_id, b_id = (self.models.model_to_id[name] for name in MODEL_NAMES)
        true = data["true_labels"]
        a = E[true == a_id]
        b = E[true == b_id]
        stats = hausdorff_statistics(a, b)
        a_unit = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        b_unit = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        normalized_stats = hausdorff_statistics(a_unit, b_unit)
        return {
            **stats,
            "hausdorff_unit_normalized": normalized_stats["hausdorff"],
            "hausdorff_p95_unit_normalized": normalized_stats["hausdorff_p95"],
        }

    def evaluate_prefix_on_trajectory(self, dataset_idx: int, t_end: int) -> dict:
        self.models.feature_module.eval()
        self.models.output_module.eval()
        self.models.time_modules.eval()
        if hasattr(self.models, "ode_models"):
            for model in self.models.ode_models.values():
                model.eval()

        x_full = np.asarray(self.dataset.x_list[dataset_idx], dtype=np.float32)
        t_full = np.asarray(self.dataset.t_list[dataset_idx], dtype=np.float32)
        true_id = int(self.dataset.true_model_labels[dataset_idx])
        labels = torch.tensor([true_id], dtype=torch.long, device=self.device)

        x_prefix = torch.as_tensor(x_full[:t_end], dtype=torch.float32, device=self.device).unsqueeze(0)
        full_t = torch.as_tensor(t_full, dtype=torch.float32, device=self.device).unsqueeze(0)
        scaled_t = self.scale_time(full_t)

        with torch.no_grad():
            states = InferenceEngine(self.models).predict_future(
                historical_x=x_prefix,
                historical_x_mask=None,
                future_t=full_t,
                scaling_stats=self.scaling_stats,
                true_model_label=labels,
            )
            pred = states[0, :, self.target_idx].detach().cpu().numpy()

            output_feature = self.models.feature_module(
                x=x_prefix, t=scaled_t, mask=None, true_model_label=labels,
            )
            e_feature = extract_embeddings(output_feature)[0].detach().cpu().numpy()

            time_module = self.get_time_module_for_label(true_id)
            e_time = time_module(t=scaled_t)[0].detach().cpu().numpy()

        clean = self.clean_target_physical(dataset_idx)
        if e_time.shape != e_feature.shape or e_feature.shape != (len(clean), EMBEDDING_DIM):
            raise ValueError(
                f"Incompatible time/feature embedding shapes: {e_time.shape}, {e_feature.shape}"
            )
        cosine = paired_cosine_similarity(e_time, e_feature)

        if np.linalg.norm(e_time - e_time.mean(axis=0)) <= 1e-12 or \
                np.linalg.norm(e_feature - e_feature.mean(axis=0)) <= 1e-12:
            procrustes_disparity = float("nan")
        else:
            _, _, procrustes_disparity = procrustes(e_time, e_feature)

        mse_future = float(np.mean((pred[t_end:] - clean[t_end:]) ** 2))

        return {
            "t_end": int(t_end),
            "predicted_trajectory": pred,
            "true_target": clean,
            "latent_time": e_time,
            "latent_feature": e_feature,
            "cosine_similarity": cosine,
            "procrustes_disparity": float(procrustes_disparity),
            "true_model_id": true_id,
            "true_model_name": self.id_to_model[true_id],
            "dataset_idx": int(dataset_idx),
            "mse_future": mse_future,
        }

    def fit_shared_pca_for_phase_results(self, phase_results: list[dict]) -> PCA:
        all_features = np.concatenate([result["latent_feature"] for result in phase_results], axis=0)
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

    def plot_multimodel_generalization_4x4(self, phase_results, shared_pca,
                                           output_path, hausdorff_metrics):
        """Same layout, colors, legend positions and scaling as single-model plot."""
        if len(phase_results) != 4:
            raise ValueError("Exactly four observation windows are required.")
        configure_plot_style()
        fig = plt.figure(figsize=(26.5, 19.0), facecolor="white")
        grid = fig.add_gridspec(
            5, 4, height_ratios=[1, 1, 1, 1, .84],
            left=.060, right=.982, top=.875, bottom=.052,
            wspace=.32, hspace=.57,
        )
        model_name = phase_results[0]["true_model_name"]
        horizon = len(phase_results[0]["true_target"])
        times = np.arange(horizon)
        actual = phase_results[0]["true_target"]

        fig.suptitle(
            f"Többmodelles EINN · járványdinamikai és látens-térbeli elemzés   |   "
            f"{model_name}   |   $d_e={EMBEDDING_DIM}$",
            fontsize=20, fontweight="bold", color=PALETTE["ink"], y=.987,
        )
        fig.text(.5, .952,
                 "Négy megfigyelési ablak · azonos teszttrajektória · "
                 "megfelelő Time Module · közös PCA-bázis",
                 ha="center", fontsize=12, color=PALETTE["muted"])

        pred_handles = [
            Line2D([0], [0], color="#171717", lw=2.6, label="Zajmentes referencia"),
            Line2D([0], [0], color=PALETTE["teal"], lw=2.6, label="Modellbecslés"),
            Patch(color=PALETTE["background_obs"], label="Megfigyelt időszak"),
        ]
        pca_handles = [
            Line2D([0], [0], lw=0, marker="x", color=PALETTE["ink"], ms=8,
                   label="Kezdet"),
            Line2D([0], [0], lw=0, marker="s", color=PALETTE["orange"], ms=8,
                   label="Végpont"),
            Line2D([0], [0], lw=0, marker="o", markerfacecolor="none",
                   markeredgecolor=PALETTE["magenta"], ms=9,
                   label="Megfigyelés vége"),
        ]
        fig.legend(handles=pred_handles, loc="upper left",
                   bbox_to_anchor=(.055, .935), ncol=3, frameon=False, fontsize=9.6)
        fig.legend(handles=pca_handles, loc="upper right",
                   bbox_to_anchor=(.968, .935), ncol=3, frameon=False, fontsize=9.6)

        predicted = np.concatenate([r["predicted_trajectory"] for r in phase_results])
        ypred = _finite_limits(np.r_[actual, predicted],
                               relative_pad=.08, min_span=.01)

        for row, r in enumerate(phase_results):
            n = int(r["t_end"])
            ax = fig.add_subplot(grid[row, 0])
            ax.plot(times, actual, color="#171717", lw=2.6, zorder=3)
            ax.plot(times, r["predicted_trajectory"],
                    color=PALETTE["teal"], lw=2.75, zorder=3)
            _mark_observed(ax, n, horizon)
            ax.set_ylim(*ypred)
            ax.set_title(f"{n} megfigyelés · jövőbeli MSE: {r['mse_future']:.3g}", pad=9)
            ax.set_xlabel("Időindex")
            ax.set_ylabel("Fertőzöttek aránya")
            _panel_style(ax, time_axis=True)

            ax = fig.add_subplot(grid[row, 1])
            _pca_axis(ax, r["cum_variance"], r["pca_95"], r["pca_99"],
                      PCA_COMPONENTS_TO_SHOW)

            ax = fig.add_subplot(grid[row, 2])
            coords = r["pca_coords"]
            ax.plot(coords[:, 0], coords[:, 1], color=PALETTE["gray"],
                    lw=1.45, alpha=.58, zorder=1)
            ax.scatter(coords[:, 0], coords[:, 1], s=16, color=PALETTE["blue"],
                       alpha=.94, edgecolors="none", zorder=2)
            ax.scatter(*coords[0], marker="x", color=PALETTE["ink"],
                       s=65, lw=1.8, zorder=4)
            ax.scatter(*coords[-1], marker="s", color=PALETTE["orange"],
                       s=67, zorder=5)
            ax.scatter(*coords[n - 1], marker="o", facecolors="none",
                       edgecolors=PALETTE["magenta"], lw=2.1, s=125, zorder=6)
            ax.set_title("Feature-látens pálya · közös PCA", pad=9)
            ax.set_xlabel(f"1. főkomponens ({shared_pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"2. főkomponens ({shared_pca.explained_variance_ratio_[1]:.1%})")
            _panel_style(ax)

            ax = fig.add_subplot(grid[row, 3])
            _cosine_axis(ax, r["cosine_similarity"], times, n, horizon)

        ax_procrustes = fig.add_subplot(grid[4, :])
        _procrustes_bars(ax_procrustes, phase_results)
        _save_plot(fig, output_path)

    def plot_joint_latent_space_hausdorff(self, data, metrics):
        """All independent held-out trajectories in ONE common PCA coordinate system.

        Hausdorff bars are computed in ORIGINAL d_e, never on the 2D projection.
        The central map shows temporal Feature curves and uses model colors.
        """
        configure_plot_style()
        E = np.asarray(data["feature_embeddings"], dtype=float)
        labels = np.asarray(data["true_labels"], dtype=int)
        trajectory_ids = np.asarray(data["trajectory_ids"], dtype=int)
        if E.ndim != 2 or E.shape[0] < 3:
            raise ValueError(f"Invalid held-out embedding matrix: {E.shape}")
        if not np.isfinite(E).all():
            raise ValueError("Joint latent PCA: non-finite embeddings")

        shared = PCA(n_components=2)
        coords = shared.fit_transform(E)
        model_a, model_b = MODEL_NAMES
        model_colors = {
            model_a: PALETTE["teal"],
            model_b: PALETTE["magenta"],
        }
        fig = plt.figure(figsize=(19.0, 11.4), facecolor="white")
        grid = fig.add_gridspec(
            nrows=3, ncols=5, left=.052, right=.963, top=.84, bottom=.105,
            width_ratios=[1.0, 1.0, 1.0, .83, .83],
            height_ratios=[.52, 2.0, 1.0],
            wspace=.68, hspace=.70,
        )
        fig.suptitle(
            f"EINN · közös Feature-látens tér és modellkülönbségek | "
            f"{model_a} / {model_b} | $d_e={EMBEDDING_DIM}$",
            fontsize=20, color=PALETTE["ink"], fontweight="bold", y=.982,
        )
        fig.text(.5, .938,
                 "Az összes független teszttrajektória · egyetlen közös PCA-bázis · "
                 "pontok időrendben összekötve",
                 fontsize=11.5, color=PALETTE["muted"], ha="center")

        # The joint PCA map is deliberately the dominant element in the figure.
        ax = fig.add_subplot(grid[:, :3])
        class_counts = {}
        for model_name in MODEL_NAMES:
            class_id = self.models.model_to_id[model_name]
            class_counts[model_name] = int(np.count_nonzero([
                int(self.dataset.true_model_labels[i]) == class_id
                for i in range(len(self.dataset.true_model_labels))
            ]))
        for model_name in MODEL_NAMES:
            class_id = self.models.model_to_id[model_name]
            color = model_colors[model_name]
            for tr in np.unique(trajectory_ids[labels == class_id]):
                ix = np.flatnonzero((labels == class_id) & (trajectory_ids == tr))
                p = coords[ix]
                ax.plot(p[:, 0], p[:, 1], color=color, lw=1.75, alpha=.47, zorder=2)
                ax.scatter(p[:, 0], p[:, 1], color=color, s=7.5,
                           alpha=.28, edgecolors="none", zorder=3, rasterized=True)
                ax.scatter(p[0, 0], p[0, 1], marker="o", color=color,
                           edgecolors="white", linewidths=.65,
                           s=33, alpha=.93, zorder=5)
                ax.scatter(p[-1, 0], p[-1, 1], marker="s", color=color,
                           edgecolors="white", linewidths=.65,
                           s=33, alpha=.93, zorder=5)
        ax.set_title("Közös látens tér · valamennyi tesztidősor", pad=14, fontsize=15)
        ax.set_xlabel(f"1. főkomponens ({shared.explained_variance_ratio_[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"2. főkomponens ({shared.explained_variance_ratio_[1]:.1%})", fontsize=12)
        ax.legend(handles=[
            Line2D([0], [0], color=model_colors[model], lw=3.1,
                   label=f"{model} · {class_counts[model]} trajektória")
            for model in MODEL_NAMES
        ], loc="upper right", frameon=True, framealpha=.95, fontsize=11)
        _panel_style(ax)
        ax.tick_params(labelsize=10)

        # Quantitative bars: each raw-space direction is a well-defined statistic.
        side = fig.add_subplot(grid[1, 3:])
        bars = np.array([metrics["directed_a_to_b"], metrics["directed_b_to_a"],
                         metrics["hausdorff_p95"]], dtype=float)
        ypos = np.arange(3)
        bars_colors = [PALETTE["teal"], PALETTE["magenta"], PALETTE["blue"]]
        side.barh(ypos, bars, height=.55, color=bars_colors, alpha=.92)
        side.set_yticks(ypos, labels=[
            f"{model_a} → {model_b}",
            f"{model_b} → {model_a}",
            "Módosított H95",
        ])
        side.invert_yaxis()
        max_val = float(max(bars.max(), 1e-8))
        side.set_xlim(0, max_val * 1.30)
        for i, v in enumerate(bars):
            side.text(v + max_val*.025, i, f"{v:.5g}", va="center", ha="left",
                      color=PALETTE["ink"], fontweight="bold", fontsize=10)
        side.set_xlabel(f"Távolság az eredeti {E.shape[1]}-dimenziós térben")
        side.set_title("Irányított Hausdorff-értékek", fontsize=12, pad=9)
        side.grid(axis="x")
        side.set_axisbelow(True)
        side.spines["left"].set_visible(False)
        side.tick_params(axis="y", length=0, labelsize=10)

        # Main headline: symmetric max of directed distances; L2 normalized separately.
        headline = fig.add_subplot(grid[0, 3:])
        headline.axis("off")
        headline.text(.5, .75, "Hausdorff-távolság", ha="center", va="center",
                      color=PALETTE["muted"], fontsize=12)
        headline.text(.5, .27, f"{metrics['hausdorff']:.6g}",
                      ha="center", va="center", color=PALETTE["ink"],
                      fontsize=30, fontweight="bold")

        note = fig.add_subplot(grid[2, 3:])
        note.axis("off")
        note.text(0, .99,
                  "Kiegészítő mértékek", color=PALETTE["ink"],
                  fontsize=12, fontweight="semibold", va="top")
        note.text(0, .75,
                  f"Egységvektor-normalizált H:  {metrics['hausdorff_unit_normalized']:.5g}\n"
                  f"Normalizált H95:  {metrics['hausdorff_p95_unit_normalized']:.5g}\n"
                  f"PCA-vetület varianciája:  {sum(shared.explained_variance_ratio_):.1%}\n"
                  f"Tesztpontok száma:  {len(E)}",
                  va="top", fontsize=10.7, linespacing=1.55,
                  color=PALETTE["muted"])
        fig.text(.5, .035,
                 "A középső ábra 2D PCA-vetület; a Hausdorff-metrikákat "
                 "az eredeti beágyazási térben számoltuk. "
                 "A színek szerinti elkülönülés önmagában nem bizonyít "
                 "címkefüggetlen modellazonosítást.",
                 ha="center", va="bottom", fontsize=9.6,
                 color=PALETTE["muted"])

        destination = os.path.join(
            self.output_dir,
            f"multimodel_kozos_latens_ter_hausdorff_{model_a.lower()}_"
            f"{model_b.lower()}_de_{EMBEDDING_DIM}.png",
        )
        _save_plot(fig, destination)

        np.savez_compressed(
            os.path.splitext(destination)[0] + ".npz",
            feature_embeddings=E,
            true_labels=labels,
            trajectory_ids=trajectory_ids,
            time_indices=np.asarray(data["time_indices"]),
            joint_pca_coords=coords,
            joint_pca_components=shared.components_,
            joint_pca_mean=shared.mean_,
            joint_pca_variance_ratio=shared.explained_variance_ratio_,
            hausdorff=float(metrics["hausdorff"]),
            hausdorff_p95=float(metrics["hausdorff_p95"]),
            hausdorff_unit=float(metrics["hausdorff_unit_normalized"]),
        )
        print("Joint latent space and Hausdorff figure:", destination)
        return destination

    def generate_multimodel_window_reports(self):
        # This is the SAME full-trajectory held-out representation used by the
        # earlier Hausdorff report, not training windows and not pooled vectors.
        all_test_data = self.collect_full_trajectories(all_trajectories=True)
        E = all_test_data["feature_embeddings"]
        truth = all_test_data["true_labels"]
        a_id, b_id = (self.models.model_to_id[name] for name in MODEL_NAMES)
        stats = hausdorff_statistics(E[truth == a_id], E[truth == b_id])
        a, b = E[truth == a_id], E[truth == b_id]
        ua = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        ub = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        unit_stats = hausdorff_statistics(ua, ub)
        hausdorff_metrics = {
            **stats,
            "hausdorff_unit_normalized": unit_stats["hausdorff"],
            "hausdorff_p95_unit_normalized": unit_stats["hausdorff_p95"],
        }
        with open(os.path.join(self.output_dir, "hausdorff_test_embeddings.csv"),
                  "w", newline="", encoding="utf-8") as handle:
            info = {
                "model_a": MODEL_NAMES[0], "model_b": MODEL_NAMES[1],
                "num_points_a": len(a), "num_points_b": len(b),
                "embedding_dim": E.shape[1], **hausdorff_metrics,
            }
            writer = csv.DictWriter(handle, fieldnames=list(info))
            writer.writeheader()
            writer.writerow(info)
        self.plot_joint_latent_space_hausdorff(all_test_data, hausdorff_metrics)
        # Additional, existing-style embedding export for reproducibility.
        np.savez_compressed(
            os.path.join(self.output_dir, "all_heldout_feature_embeddings.npz"),
            **all_test_data,
        )

        selected = []
        for true_id in self.class_ids:
            indices = [i for i, tr in enumerate(self.dataset.true_model_labels)
                       if int(tr) == true_id]
            if not indices:
                raise RuntimeError(f"No held-out trajectory for model id {true_id}.")
            selected.append(indices[0])

        for dataset_idx in selected:
            phase_results = []
            for t_end in OBSERVATION_WINDOWS:
                phase_results.append(self.evaluate_prefix_on_trajectory(
                    dataset_idx=dataset_idx, t_end=t_end,
                ))
            shared_pca = self.fit_shared_pca_for_phase_results(phase_results)
            model_name = phase_results[0]["true_model_name"]
            prefix = (f"multimodel_generalization_{'_'.join(m.lower() for m in MODEL_NAMES)}_"
                      f"true_{model_name.lower()}_de_{EMBEDDING_DIM}_4x4")
            output_path = os.path.join(self.output_dir, prefix + ".png")
            self.plot_multimodel_generalization_4x4(
                phase_results=phase_results,
                shared_pca=shared_pca,
                output_path=output_path,
                hausdorff_metrics=hausdorff_metrics,
            )
            np.savez_compressed(
                os.path.join(self.output_dir, prefix + ".npz"),
                observation_windows=np.asarray(OBSERVATION_WINDOWS),
                true_target=phase_results[0]["true_target"],
                predictions=np.stack([r["predicted_trajectory"] for r in phase_results]),
                time_embeddings=np.stack([r["latent_time"] for r in phase_results]),
                feature_embeddings=np.stack([r["latent_feature"] for r in phase_results]),
                cosine_similarities=np.stack([r["cosine_similarity"] for r in phase_results]),
                pca_coords=np.stack([r["pca_coords"] for r in phase_results]),
                shared_pca_variance_ratio=shared_pca.explained_variance_ratio_,
                cumulative_pca_variance=np.stack([r["cum_variance"] for r in phase_results]),
                pca_d95=np.asarray([r["pca_95"] for r in phase_results]),
                pca_d99=np.asarray([r["pca_99"] for r in phase_results]),
                procrustes=np.asarray([r["procrustes_disparity"] for r in phase_results]),
                future_mse=np.asarray([r["mse_future"] for r in phase_results]),
            )
            with open(os.path.join(self.output_dir, prefix + "_metrics.csv"),
                      "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["observed_steps", "future_mse", "pca_d95", "pca_d99",
                                 "procrustes_disparity", "mean_cosine"])
                for r in phase_results:
                    writer.writerow([r["t_end"], r["mse_future"], r["pca_95"],
                                     r["pca_99"], r["procrustes_disparity"],
                                     float(np.nanmean(r["cosine_similarity"]))])
            print("4x4 generalization figure saved:", output_path)

    def generate_all_reports(self):
        self.generate_multimodel_window_reports()


# =====================================================================
# EXPERIMENT ENTRY POINT
# =====================================================================

def reset_seeds():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def run_experiment():
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


if __name__ == "__main__":
    run_experiment()
