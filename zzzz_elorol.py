"""Train and evaluate the SIR/SEIR multimodel EINN experiments sequentially.

One GPU (cuda:0) is used for one experiment at a time:
SIR/SEIR with d_e=5, followed by SIR/SEIR with d_e=20.
Fresh synthetic train/test trajectories; Feature Module conditioned on true labels.
The silhouette-score is computed in original d_e dimensions; PCA is for display.

Run without arguments from the project root:
    python -u einn_sir_seir_silhouette_only.py
"""

import gc
import csv
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from einn_multimodel.config.einn_model_config import EINNModelConfig
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.data.multi_model_input_loader import MultiModelInputLoader
from einn_multimodel.inference.training_post_processor import TrainingPostProcessor
from einn_multimodel.inference.inference_engine import InferenceEngine
from einn_multimodel.model.einn_builder import EINNBuilder
from einn_multimodel.training.einn_trainer import EINNTrainer


SEED = 42
MODEL_NAMES = ("SIR", "SEIR")  # Set automatically for each experiment.
TRAIN_STATES_PER_MODEL = 100
TEST_STATES_PER_MODEL = 20
SEQ_LEN = 200
NOISE_STD = 0.01
EPOCHS = 5
BATCH_SIZE = 32
DEVICE = "cuda:1"  # main() selects cuda:0, or CPU if CUDA is unavailable.
# Set to (False,) or (True,) to run only one variant.
# (False, True) runs both with identical held-out trajectories.
NORMALIZATION_MODES = (False,)
# A: (True,) feeds the true label to Feature twice, as before.
# B: (False,) feeds the SAME learned label token to all samples.
# (True, False) runs both controls with identical training/test trajectories.
FEATURE_LABEL_CONDITIONING_MODES = (True,)
EMBEDDING_DIM = 20   # Set automatically for each experiment.
OUTPUT_ROOT = "./outputs/multimodel_silhouette_runs"

PARAM_CALIBRATION = {
    "SIR": {"beta": 0.4, "gamma": 0.1},
    "SIS": {"beta": 0.4, "gamma": 0.1},
    "SEIR": {"beta": 0.4, "alpha": 0.2, "gamma": 0.1},
}
MODEL_PARAM_RANGES = {
    "SIR": {"beta": (0.0, 1.0), "gamma": (0.0, 1.0)},
    "SIS": {"beta": (0.0, 1.0), "gamma": (0.0, 1.0)},
    "SEIR": {
        "beta": (0.0, 1.0),
        "alpha": (0.0, 1.0),
        "gamma": (0.0, 1.0),
    },
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
# KÖZÖS LÁTENS TÉR — A MENTETT EREDMÉNYEK ÚJRARAJZOLÓJÁVAL AZONOS STÍLUS
# =====================================================================

POINT_SIZE = 30         # scatter-pontok területe (pont^2)
POINT_ALPHA = 0.46
CURVE_WIDTH = 1.4
CURVE_ALPHA = 0.42
ENDPOINT_SIZE = 55
JOINT_FIGSIZE = (15.5, 10.0)
JOINT_PNG_DPI = 260
JOINT_OUTPUT_SUBDIR = "ujrarajzolt_kozos_latens_ter"

JOINT_PALETTE = {
    "ink": "#18283D",
    "muted": "#607185",
    "grid": "#D8E1EB",
    "teal": "#008E97",
    "magenta": "#AB4C77",
}


def configure_joint_style():
    """A külön újrarajzolóban megadott betűméretek, színek és PDF-fontok."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#AAB8C8",
        "axes.labelcolor": JOINT_PALETTE["ink"],
        "text.color": JOINT_PALETTE["ink"],
        "xtick.color": JOINT_PALETTE["muted"],
        "ytick.color": JOINT_PALETTE["muted"],
        "grid.color": JOINT_PALETTE["grid"],
        "grid.alpha": 0.8,
        "grid.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


class FeatureOnlyEvaluator:
    """Only evaluate Feature embeddings and predictions on the held-out set."""

    def __init__(self, models, dataset, metrics, scaling_stats, device, output_dir, target_scaler):
        self.models = models
        self.dataset = dataset
        self.metrics = metrics
        self.scaling_stats = scaling_stats
        # y_list stores y_scaler.transform(clean_y), not the physical clean_y.
        # The training-fitted y_scaler is essential for inverse transforming it.
        if target_scaler is None or not hasattr(target_scaler, "inverse_transform"):
            raise ValueError("Pass the training-fitted data_loader.preprocessor.y_scaler.")
        self.target_scaler = target_scaler
        target_indices = {model.target_idx for model in models.ode_models.values()}
        if len(target_indices) != 1:
            raise ValueError("All models must agree on the target compartment index.")
        self.target_idx = next(iter(target_indices))
        aux_mean = float(scaling_stats.aux_mean[self.target_idx])
        aux_std = float(scaling_stats.aux_std[self.target_idx])
        y_mean = float(self.target_scaler.mean_[0])
        y_std = float(self.target_scaler.scale_[0])
        # The synthetic data loader anchors aux[:, target_idx] to clean y.
        # This check detects train/inference scaler inconsistencies.
        if not np.allclose([aux_mean, aux_std], [y_mean, y_std], rtol=1e-3, atol=1e-6):
            raise ValueError(
                "Incompatible y and aux target scaling! "
                f"y=({y_mean}, {y_std}), aux=({aux_mean}, {aux_std}). "
                "Check model target_idx and preprocessing before evaluating."
            )
        self.device = device
        self.output_dir = output_dir
        self.id_to_model = {v: k for k, v in models.model_to_id.items()}
        self.class_ids = sorted(self.id_to_model)
        os.makedirs(output_dir, exist_ok=True)

    def scale_time(self, t):
        minimum = self.scaling_stats.t_min.to(self.device)
        maximum = self.scaling_stats.t_max.to(self.device)
        return (t - minimum) / (maximum - minimum).clamp_min(1e-8)

    def clean_target_physical(self, index: int) -> np.ndarray:
        """Inverse-transform held-out CLEAN y to the compartment's physical scale."""
        scaled = np.asarray(self.dataset.y_list[index], dtype=np.float64).reshape(-1, 1)
        clean = self.target_scaler.inverse_transform(scaled).reshape(-1)
        if not np.isfinite(clean).all():
            raise ValueError(f"Non-finite clean target at dataset index {index}.")
        return clean

    def verify_target_scaling(self):
        """Ensure the plotted target is not mistaken for a standardized value."""
        scaled = np.concatenate([
            np.asarray(y).reshape(-1) for y in self.dataset.y_list
        ])
        clean = np.concatenate([
            self.clean_target_physical(i) for i in range(len(self.dataset.y_list))
        ])
        print(
            "Test target scaling check | "
            f"scaled range=({scaled.min():.5g}, {scaled.max():.5g}) | "
            f"physical range=({clean.min():.5g}, {clean.max():.5g}) | "
            f"physical initial sample={self.clean_target_physical(0)[0]:.5g}"
        )
        if clean.min() < -1e-5:
            print("WARNING: Physically negative target values; inspect the ODE generator.")

    @staticmethod
    def component(metric, name):
        if isinstance(metric, dict):
            return (metric.get("loss_components") or {}).get(name, np.nan)
        return (getattr(metric, "loss_components", None) or {}).get(name, np.nan)

    def plot_losses(self):
        """A teljes tanítás súlyozott veszteségei, fázisonként külön ábrázolva.

        A Phase 3 nem számít ode_F és ode_future_F tagot; ezek a Phase 4-ben
        aktívak. A hiányzó komponenseket nem nullával helyettesítjük.
        """
        loss_labels = {
            "ode_T": "Time-modul: ODE-illeszkedés",
            "ode_future_T": "Time-modul: jövőbeli ODE-illeszkedés",
            "ode_F": "Feature-modul: ODE-illeszkedés",
            "ode_future_F": "Feature-modul: jövőbeli ODE-illeszkedés",
            "aux": "Segédadatokhoz illeszkedés (aux)",
            "data_T": "Time-modul: adatilleszkedés",
            "data_F": "Feature-modul: adatilleszkedés",
            "kd_target": "Tudáslepárlás: állapotbecslések",
            "kd_emb": "Tudáslepárlás: beágyazások",
            "ic": "Time-modul: kezdeti feltétel",
            "ic_F": "Feature-modul: kezdeti feltétel",
            "mono": "Monotonitási feltétel",
            "param": "ODE-paraméterek simasága",
            "supcon": "Felügyelt kontrasztív veszteség (SupCon)",
        }

        # Egyetlen CSV-ben minden fázis adatai is megmaradnak.
        csv_path = os.path.join(self.output_dir, "training_loss_components.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["lepes", "fazis"] + list(loss_labels))
            for step, metric in enumerate(self.metrics, 1):
                phase = (metric.get("phase") if isinstance(metric, dict)
                         else getattr(metric, "phase", None))
                row = [step, phase]
                for name in loss_labels:
                    value = self.component(metric, name)
                    row.append(float(value) if value is not None and np.isfinite(value) else "")
                writer.writerow(row)

        for phase in (1, 2, 3, 4):
            phase_metrics = [
                m for m in self.metrics
                if (m.get("phase") if isinstance(m, dict)
                    else getattr(m, "phase", None)) == phase
            ]
            if not phase_metrics:
                continue

            # Csak az adott fázisban ténylegesen számított veszteségek.
            available = []
            for name in loss_labels:
                values = np.asarray(
                    [self.component(m, name) for m in phase_metrics], dtype=float
                )
                if np.isfinite(values).any():
                    available.append((name, values))

            if not available:
                print(f"A(z) {phase}. fázisban nincs naplózott veszteség.")
                continue

            ncols = 3
            nrows = (len(available) + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(17, max(3.7 * nrows, 5)),
                squeeze=False,
                constrained_layout=True,
            )
            fig.suptitle(
                f"EINN tanítás – {phase}. fázis | súlyozott veszteségfüggvények",
                fontsize=16, fontweight="bold",
            )
            steps = np.arange(1, len(phase_metrics) + 1)
            for ax, (name, values) in zip(axes.flat, available):
                valid = np.isfinite(values)
                color = ("#bc6034" if name.startswith("ode_") else
                         "#2563a6" if name.startswith("kd_") else "#177e74")
                ax.plot(steps[valid], values[valid], lw=1.7, color=color)
                ax.set_title(f"{loss_labels[name]}  [{name}]", fontsize=10)
                ax.set_xlabel("Optimalizációs lépés")
                ax.set_ylabel("Súlyozott veszteség")
                ax.grid(alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
            for ax in list(axes.flat)[len(available):]:
                ax.axis("off")

            filename = os.path.join(
                self.output_dir, f"training_loss_curves_phase_{phase}.png"
            )
            fig.savefig(filename, dpi=230, bbox_inches="tight")
            # A korábbi fájlnév is megmarad a 3. fázis ábrájának.
            if phase == 3:
                fig.savefig(
                    os.path.join(self.output_dir, "training_loss_curves.png"),
                    dpi=230, bbox_inches="tight",
                )
            plt.close(fig)
            print(f"Veszteséggörbék mentve: {filename}")

    def collect_full_trajectories(self, all_trajectories: bool = False):
        """Collect Feature embeddings of held-out, correctly labelled trajectories.

        Output shape [total_valid_timesteps, d_e]; NO pooling over time.
        """
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
                x = torch.as_tensor(
                    self.dataset.x_list[idx], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                t = torch.as_tensor(
                    self.dataset.t_list[idx], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                true_id = int(self.dataset.true_model_labels[idx])
                labels = torch.tensor([true_id], dtype=torch.long, device=self.device)
                e_f = self.models.feature_module(
                    x=x, t=self.scale_time(t), mask=None,
                    true_model_label=labels,
                ).embeddings.squeeze(0).cpu().numpy()
                if e_f.ndim != 2:
                    raise ValueError(f"Expected [T,d_e], got {e_f.shape}.")
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

    def plot_latent_space(self, data):
        """Plot all held-out Feature trajectories using one common PCA basis.

        Silhouette is computed in the original d_e-dimensional embedding space.
        """
        E = np.asarray(data["feature_embeddings"], dtype=np.float64)
        labels = np.asarray(data["true_labels"], dtype=int)
        trajectory_ids = np.asarray(data["trajectory_ids"], dtype=int)
        time_indices = np.asarray(data["time_indices"], dtype=int)
        pair = tuple(name.upper() for name in MODEL_NAMES)
        model_ids = self.models.model_to_id
        d_e = E.shape[1]

        if E.ndim != 2 or d_e < 2 or len(E) < 3:
            raise ValueError(f"Nem megfelelő tesztbeágyazások: {E.shape}")
        if not np.isfinite(E).all():
            raise ValueError("A tesztbeágyazások NaN/Inf értékeket tartalmaznak.")
        if any(len(v) != len(E) for v in (labels, trajectory_ids, time_indices)):
            raise ValueError("Az embeddingek és a címkék hossza eltér.")
        expected_labels = {int(model_ids[name]) for name in pair}
        if set(np.unique(labels).tolist()) != expected_labels:
            raise ValueError("A tesztcímkék nem egyeznek a kiválasztott modellpárral.")

        _, n_per_model = np.unique(labels, return_counts=True)
        if len(n_per_model) != 2 or np.any(n_per_model < 2):
            raise ValueError("Silhouette requires two labels with at least two points each.")
        silhouette = float(silhouette_score(E, labels, metric="euclidean"))
        fitted = PCA(n_components=2, svd_solver="full")
        coords = fitted.fit_transform(E)
        variance = fitted.explained_variance_ratio_
        colors = {pair[0]: JOINT_PALETTE["teal"], pair[1]: JOINT_PALETTE["magenta"]}

        # A stílus csak erre a diagramra hasson; a veszteség- és előrejelzési
        # ábrák matplotlib-beállításait ne módosítsuk.
        with plt.rc_context():
            configure_joint_style()
            fig, ax = plt.subplots(figsize=JOINT_FIGSIZE)
            fig.subplots_adjust(left=.095, right=.97, bottom=.10, top=.80)
            fig.suptitle(
                f"EINN · közös látens tér   |   {pair[0]} / {pair[1]}   |   $d_e={d_e}$",
                fontsize=19, fontweight="bold", color=JOINT_PALETTE["ink"], y=.965,
            )
            fig.text(
                .5, .912,
                "Az összes teszttrajektória · egyetlen közös PCA-bázis · pontok időrendben összekötve",
                fontsize=11.4, color=JOINT_PALETTE["muted"], ha="center",
            )
            fig.text(
                .5, .864,
                rf"Silhouette-score: $S={silhouette:.4f}$",
                fontsize=20.6, color=JOINT_PALETTE["ink"], ha="center",
            )

            counts = {}
            for name in pair:
                class_id = model_ids[name]
                class_trajectory_ids = np.unique(trajectory_ids[labels == class_id])
                counts[name] = len(class_trajectory_ids)
                for tid in class_trajectory_ids:
                    indices = np.flatnonzero(
                        (trajectory_ids == tid) & (labels == class_id)
                    )
                    indices = indices[np.argsort(time_indices[indices], kind="stable")]
                    p = coords[indices]
                    if len(p) < 1:
                        continue
                    ax.plot(p[:, 0], p[:, 1], color=colors[name],
                            lw=CURVE_WIDTH, alpha=CURVE_ALPHA, zorder=2)
                    ax.scatter(p[:, 0], p[:, 1], color=colors[name],
                               s=POINT_SIZE, alpha=POINT_ALPHA,
                               edgecolors="none", zorder=3, rasterized=True)
                    # Az általad módosított jelölések: kezdet=négyzet, vég=háromszög.
                    ax.scatter(*p[0], marker="s", s=ENDPOINT_SIZE, color=colors[name],
                               edgecolors="white", linewidths=.7, zorder=5)
                    ax.scatter(*p[-1], marker="^", s=ENDPOINT_SIZE, color=colors[name],
                               edgecolors="white", linewidths=.7, zorder=5)

            ax.set_title("Valamennyi tesztidősor látens pályája", fontsize=14, pad=12)
            ax.set_xlabel(f"1. főkomponens ({variance[0]:.1%})", fontsize=20)
            ax.set_ylabel(f"2. főkomponens ({variance[1]:.1%})", fontsize=20)
            ax.grid(True, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=10)
            ax.legend(handles=[
                Line2D([0], [0], color=colors[name], lw=2.7, marker="o",
                       markersize=6, label=f"{name} · {counts[name]} trajektória")
                for name in pair
            ] + [
                Line2D([0], [0], color=JOINT_PALETTE["ink"], lw=0,
                       marker="s", markersize=7, label="Kezdet"),
                Line2D([0], [0], color=JOINT_PALETTE["ink"], lw=0,
                       marker="^", markersize=7, label="Végpont"),
            ], loc="best", frameon=True, framealpha=.94, fontsize=18)

            fig.text(
                .5, .031,
                "A silhouette-score-t az eredeti $d_e$-dimenziós térben számítottuk, nem a PCA-vetületben.",
                ha="center", color=JOINT_PALETTE["muted"], fontsize=10,
            )

            # A korábbi név alatt is az ÚJ ábra jelenjen meg, és a független
            # újrarajzolóval azonos kimeneti néven PNG/PDF is készüljön.
            out_dir = os.path.join(self.output_dir, JOINT_OUTPUT_SUBDIR)
            os.makedirs(out_dir, exist_ok=True)
            stem = f"kozos_latens_ter_silhouette_{pair[0].lower()}_{pair[1].lower()}_de_{d_e}"
            png = os.path.join(out_dir, stem + ".png")
            pdf = os.path.join(out_dir, stem + ".pdf")
            fig.savefig(png, dpi=JOINT_PNG_DPI, bbox_inches="tight", facecolor="white")
            fig.savefig(pdf, bbox_inches="tight", facecolor="white")
            fig.savefig(os.path.join(self.output_dir, "feature_latent_space_pca.png"),
                        dpi=JOINT_PNG_DPI, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        with open(os.path.join(out_dir, stem + "_meroszamok.csv"),
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model_a", "model_b", "d_e", "silhouette_score",
                             "pc1_variance", "pc2_variance", "n_test_points",
                             "n_trajectories_a", "n_trajectories_b"])
            writer.writerow([pair[0], pair[1], d_e, silhouette,
                             variance[0], variance[1], len(E),
                             counts[pair[0]], counts[pair[1]]])

        # A korábbi kód e kiértékelési fájljait továbbra is megtartjuk.
        np.savez_compressed(
            os.path.join(self.output_dir, "feature_embeddings_test.npz"), **data
        )
        lines = [
            "=== HELD-OUT FEATURE EMBEDDING GEOMETRY ===",
            f"PCA variance PC1: {variance[0]:.8f}",
            f"PCA variance PC2: {variance[1]:.8f}",
            f"Feature embedding normalization: {self.models.feature_module.normalize_embeddings}",
            f"Mean embedding norm: {np.linalg.norm(E, axis=1).mean():.6f}",
            f"Held-out trajectories: {len(data['selected_dataset_indices'])}",
        ]
        result = {
            "mean_embedding_norm": float(np.linalg.norm(E, axis=1).mean()),
            "silhouette_test": silhouette,
        }
        lines.append(f"Silhouette (original d_e): {silhouette:.6f}")
        lines.append("Temporal points of the same trajectory are NOT independent samples.")
        report = "\n".join(lines)
        print("\n" + report)
        with open(os.path.join(self.output_dir, "feature_metrics.txt"), "w") as f:
            f.write(report)
        print(f"Közös PCA-ábra elmentve: {png}; {pdf}")
        return result

    def evaluate_heldout_future_mse(self):
        """Future-only test MSE in PHYSICAL units from an 80-step prefix.

        predict_future() returns compartment predictions after inverse AUX scaling.
        The dataset's clean y_list, however, is standardized with Y scaling.
        Invert y_list before comparing; otherwise the MSE is meaningless.
        """
        engine = InferenceEngine(self.models)
        prefix = min(80, SEQ_LEN - 1)
        errors = []
        group_errors = {}
        self.models.feature_module.eval()
        self.models.output_module.eval()
        with torch.no_grad():
            for idx in range(len(self.dataset.x_list)):
                x = torch.as_tensor(
                    self.dataset.x_list[idx], dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                t = torch.as_tensor(
                    self.dataset.t_list[idx], dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                if t.shape[1] <= prefix:
                    continue
                true = int(self.dataset.true_model_labels[idx])
                states = engine.predict_future(
                    historical_x=x[:, :prefix],
                    historical_x_mask=None,
                    future_t=t,
                    scaling_stats=self.scaling_stats,
                    true_model_label=torch.tensor(
                        [true], device=self.device, dtype=torch.long,
                    ),
                )
                predicted = states[0, prefix:, self.target_idx].cpu().numpy()
                clean = self.clean_target_physical(idx)
                actual = clean[prefix:prefix + len(predicted)]
                if actual.shape != predicted.shape:
                    raise ValueError("Forecast and clean reference lengths disagree.")
                err = float(np.mean((predicted - actual) ** 2))
                errors.append(err)
                key = self.id_to_model[true]
                group_errors.setdefault(key, []).append(err)

        if not errors:
            raise RuntimeError("No held-out trajectories long enough for forecast MSE.")
        future_mse = float(np.mean(errors))
        with open(os.path.join(self.output_dir, "heldout_future_mse.txt"), "w") as f:
            f.write(
                f"Held-out future MSE in PHYSICAL units, prefix={prefix}: "
                f"{future_mse:.9g}\n"
            )
            for true, values in sorted(group_errors.items()):
                f.write(f"True={true}: {np.mean(values):.9g}\n")
        print(f"Held-out future MSE (physical units, prefix={prefix}): {future_mse:.8g}")
        return future_mse

    def plot_prediction_examples(self):
        """One held-out prediction per TRUE class, with the correct model label."""
        prefix_steps = min(80, SEQ_LEN - 1)
        true_ids = self.dataset.true_model_labels
        selected = []
        for true_id in self.class_ids:
            indices = [i for i, tr in enumerate(true_ids) if int(tr) == true_id]
            if not indices:
                raise RuntimeError(f"No test trajectory for model {true_id}.")
            selected.append(indices[0])

        engine = InferenceEngine(self.models)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), squeeze=False)
        with torch.no_grad():
            for ax, idx in zip(axes.ravel(), selected):
                x = torch.as_tensor(
                    self.dataset.x_list[idx], dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                t = torch.as_tensor(
                    self.dataset.t_list[idx], dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                true_id = int(true_ids[idx])
                labels = torch.tensor([true_id], dtype=torch.long, device=self.device)
                states = engine.predict_future(
                    historical_x=x[:, :prefix_steps],
                    historical_x_mask=None,
                    future_t=t,
                    scaling_stats=self.scaling_stats,
                    true_model_label=labels,
                )
                predicted = states[0, :, self.target_idx].cpu().numpy()
                clean = self.clean_target_physical(idx)
                time = np.asarray(self.dataset.t_list[idx]).reshape(-1)
                ax.plot(time, clean, label="Clean reference (physical)",
                        color="black", alpha=0.8)
                ax.plot(time, predicted, label="Feature prediction (physical)",
                        linewidth=1.6)
                ax.axvline(time[prefix_steps], linestyle="--", color="gray",
                           label="Forecast starts")
                ax.set_ylabel("Target compartment proportion")
                ax.set_xlabel("Time index")
                ax.set_title(f"True {self.id_to_model[true_id]} | Correct model")
                ax.grid(alpha=0.3)
        axes[0, 0].legend()
        fig.suptitle("Held-out forecasts (true-model conditioning)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "heldout_prediction_examples.png"), dpi=250)
        plt.close(fig)

    def generate_all_reports(self):
        self.verify_target_scaling()
        self.plot_losses()
        # Full trajectories from held-out data only: no re-use of training windows.
        data = self.collect_full_trajectories(all_trajectories=True)
        np.savez_compressed(
            os.path.join(self.output_dir, "all_heldout_feature_embeddings.npz"), **data
        )
        embedding_metrics = self.plot_latent_space(data)
        future_mse = self.evaluate_heldout_future_mse()
        self.plot_prediction_examples()
        print(f"Saved results to: {os.path.abspath(self.output_dir)}")
        return {
            **embedding_metrics,
            "heldout_future_mse": future_mse,
        }


def reset_seeds():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def run_experiment():
    logging.basicConfig(level=logging.INFO)
    reset_seeds()
    names = tuple(name.upper() for name in MODEL_NAMES)
    if set(names) not in ({"SIR", "SIS"}, {"SIR", "SEIR"}):
        raise ValueError("Choose MODEL_NAMES = ('SIR','SIS') or ('SIR','SEIR').")
    is_sir_sis = set(names) == {"SIR", "SIS"}
    d_s = 3 if is_sir_sis else 4
    d_p = 2 if is_sir_sis else 3
    initial_states = (
        [0.99, 0.01, 0.0] if is_sir_sis else [0.99, 0.0, 0.01, 0.0]
    )

    train_config = EINNTrainConfig(
        device=DEVICE,
        learning_rate=1e-3,
        supcon_temperature=0.1,
        mask_mismatched_losses=False,  # True-model-only: all EINN losses active.
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
    print(f"Train windows: {len(train_dataset)} | Held-out test windows: {len(test_dataset)}")
    if not all(int(a) == int(b) for a, b in zip(
        train_dataset.true_model_labels, train_dataset.assumed_model_labels
    )):
        raise RuntimeError("Training dataset unexpectedly contains mismatched labels.")
    if not all(int(a) == int(b) for a, b in zip(
        test_dataset.true_model_labels, test_dataset.assumed_model_labels
    )):
        raise RuntimeError("Test dataset unexpectedly contains mismatched labels.")
    print("True-model-only routing; mismatch masking disabled.")

    results = []
    for condition_feature_on_true_label in FEATURE_LABEL_CONDITIONING_MODES:
        for normalize_feature in NORMALIZATION_MODES:
            # Reinitialize all model parameters and the sampler RNG identically.
            # The generated training/test trajectories and the scaling stats above
            # are reused unchanged by both variants.
            reset_seeds()
            variant = "l2_normalized" if normalize_feature else "raw"
            conditioning = "label_conditioned" if condition_feature_on_true_label else "no_feature_label"
            output_dir = os.path.join(OUTPUT_ROOT, conditioning, variant)
            print(f"\n=== FEATURE EMBEDDING VARIANT: {variant} ===")
            model_config = EINNModelConfig(
                d_x=5, d_e=EMBEDDING_DIM, d_s=d_s, d_p=d_p,
                feature_rnn_out=40, feature_bidirectional=True,
                num_models=2, model_label_dim=5,
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
            summary = evaluator.generate_all_reports()
            summary["normalize_feature_embeddings"] = normalize_feature
            summary["condition_feature_on_true_label"] = condition_feature_on_true_label
            results.append(summary)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "training_summary.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\n=== TRUE-LABEL-ONLY EXPERIMENT COMPARISON ===")
    for result in results:
        print(result)


# Only the two remaining SIR/SEIR experiments, sequentially on one GPU.
EXPERIMENT_GRID = (
    (("SIR", "SEIR"), 5),
    (("SIR", "SEIR"), 20),
)
OUTPUT_BASE = "./outputs/multimodel_silhouette_runs"


def main():
    global MODEL_NAMES, EMBEDDING_DIM, DEVICE, OUTPUT_ROOT

    # A single GPU is used sequentially; never create competing GPU jobs.
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE} | {len(EXPERIMENT_GRID)} sequential experiments", flush=True)

    for experiment_index, (pair, embedding_dim) in enumerate(EXPERIMENT_GRID, start=1):
        MODEL_NAMES = pair
        EMBEDDING_DIM = embedding_dim
        pair_slug = "_".join(name.lower() for name in pair)
        OUTPUT_ROOT = os.path.join(OUTPUT_BASE, f"{pair_slug}_de_{embedding_dim}")

        print("\n" + "=" * 76, flush=True)
        print(f"EXPERIMENT {experiment_index}/{len(EXPERIMENT_GRID)}: "
              f"{' / '.join(pair)}, d_e={embedding_dim}", flush=True)
        print(f"Results: {os.path.abspath(OUTPUT_ROOT)}", flush=True)
        print("=" * 76, flush=True)

        # run_experiment resets random seeds and trains a fresh model each time.
        run_experiment()

        # Release Python references / cached GPU memory before next experiment.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Completed {experiment_index}/{len(EXPERIMENT_GRID)}: "
              f"{pair_slug}, d_e={embedding_dim}", flush=True)

    print("\nBoth SIR/SEIR experiments completed successfully.", flush=True)


if __name__ == "__main__":
    main()
