"""True-model-only two-Time/one-Feature EINN experiment.

Each generated trajectory is used ONCE with its correct ODE/Time Module.
The known TRUE model label conditions FeatureModule (optionally hidden in a control).
SupCon acts on timestep-level e_t^F; OutputModule receives that same tensor.
No projection head, classifier, cross-assumption copies or loss masking.
"""

import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from einn_multimodel.config.einn_model_config import EINNModelConfig
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.data.multi_model_input_loader import MultiModelInputLoader
from einn_multimodel.inference.training_post_processor import TrainingPostProcessor
from einn_multimodel.inference.inference_engine import InferenceEngine
from einn_multimodel.model.einn_builder import EINNBuilder
from einn_multimodel.training.einn_trainer import EINNTrainer


# A mostani kísérletnél a supcon az 5 és a kd-k a 8

####################

# MAJD NÉZZÜK MEG SUPCON NÉLKÜL - EINNLOSSBAN KELL KIKOMMENTELNI

# MOST D_E=20


######################


SEED = 42
MODEL_NAMES = ("SIR", "SEIR")  # Or ("SIR", "SEIR")
TRAIN_STATES_PER_MODEL = 100
TEST_STATES_PER_MODEL = 4
SEQ_LEN = 200
NOISE_STD = 0.01
EPOCHS = 5
BATCH_SIZE = 32
DEVICE = "cuda:1"  # Change to "cuda:0" or "cpu" if needed.
# Set to (False,) or (True,) to run only one variant.
# (False, True) runs both with identical held-out trajectories.
NORMALIZATION_MODES = (False,)
# A: (True,) feeds the true label to Feature twice, as before.
# B: (False,) feeds the SAME learned label token to all samples.
# (True, False) runs both controls with identical training/test trajectories.
FEATURE_LABEL_CONDITIONING_MODES = (True,)
OUTPUT_ROOT = f"./outputs/{'_'.join(n.lower() for n in MODEL_NAMES)}_D_E_5"

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

    def plot_latent_space(self):
        """Joint PCA for ALL held-out trajectories in original d_e dimension."""
        data = self.collect_full_trajectories(all_trajectories=True)
        E = data["feature_embeddings"]
        fitted = PCA(n_components=2, random_state=SEED)
        coords = fitted.fit_transform(E)

        import pandas as pd
        frame = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "true": [self.id_to_model[i] for i in data["true_labels"]],
        })
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            data=frame, x="PC1", y="PC2", hue="true",
            s=18, alpha=0.50, ax=ax,
        )
        for trajectory_id in np.unique(data["trajectory_ids"]):
            indices = np.where(data["trajectory_ids"] == trajectory_id)[0]
            ax.plot(coords[indices, 0], coords[indices, 1],
                    alpha=0.23, linewidth=0.7, color="gray", zorder=0)
        ax.set_title(
            "Feature Module output embedding $e_t^F$ "
            f"({'L2 normalized' if self.models.feature_module.normalize_embeddings else 'raw'})\n"
            f"PC1={100*fitted.explained_variance_ratio_[0]:.1f}% | "
            f"PC2={100*fitted.explained_variance_ratio_[1]:.1f}%"
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "feature_latent_space_pca.png"), dpi=250)
        plt.close(fig)

        np.savez_compressed(
            os.path.join(self.output_dir, "feature_embeddings_test.npz"), **data
        )
        lines = [
            "=== HELD-OUT FEATURE EMBEDDING GEOMETRY ===",
            f"PCA variance PC1: {fitted.explained_variance_ratio_[0]:.8f}",
            f"PCA variance PC2: {fitted.explained_variance_ratio_[1]:.8f}",
            f"Feature embedding normalization: {self.models.feature_module.normalize_embeddings}",
            f"Mean embedding norm: {np.linalg.norm(E, axis=1).mean():.6f}",
            f"Held-out trajectories: {len(data['selected_dataset_indices'])}",
        ]
        result = {
            "mean_embedding_norm": float(np.linalg.norm(E, axis=1).mean()),
            "silhouette_test": float("nan"),
            "davies_bouldin_test": float("nan"),
        }
        if 1 < len(np.unique(data["true_labels"])) < len(E):
            try:
                sil = silhouette_score(E, data["true_labels"])
                db = davies_bouldin_score(E, data["true_labels"])
                result["silhouette_test"] = float(sil)
                result["davies_bouldin_test"] = float(db)
                lines.extend([f"Silhouette: {sil:.6f}", f"Davies-Bouldin: {db:.6f}"])
            except ValueError as exc:
                lines.append(f"Clustering metrics unavailable: {exc}")
        lines.append("Temporal points of the same trajectory are NOT independent samples.")
        report = "\n".join(lines)
        print("\n" + report)
        with open(os.path.join(self.output_dir, "feature_metrics.txt"), "w") as f:
            f.write(report)
        return result

    @staticmethod
    def hausdorff_statistics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        """Exact Euclidean Hausdorff using nearest neighbors in the original d_e-space.

        Also returns a 95th-percentile symmetric modified Hausdorff statistic
        to gauge sensitivity to a few extreme points.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError("Expected [N,d_e] and [M,d_e] with matching dimensions.")
        if len(a) == 0 or len(b) == 0:
            raise ValueError("Hausdorff requires two nonempty point sets.")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Hausdorff point sets contain non-finite coordinates.")
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

    def evaluate_hausdorff_test_embeddings(self):
        """Hausdorff in original d_e space, across ALL independent test trajectories."""
        data = self.collect_full_trajectories(all_trajectories=True)
        E = data["feature_embeddings"]
        true = data["true_labels"]
        a_id, b_id = (self.models.model_to_id[name] for name in MODEL_NAMES)
        a = E[true == a_id]
        b = E[true == b_id]
        stats = self.hausdorff_statistics(a, b)
        a_unit = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        b_unit = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        normalized_stats = self.hausdorff_statistics(a_unit, b_unit)
        record = {
            "true_model_a": self.id_to_model[a_id],
            "true_model_b": self.id_to_model[b_id],
            "num_points_a": int(len(a)),
            "num_points_b": int(len(b)),
            "embedding_dim": int(E.shape[1]),
            **stats,
            "hausdorff_unit_normalized": normalized_stats["hausdorff"],
            "hausdorff_p95_unit_normalized": normalized_stats["hausdorff_p95"],
        }
        with open(os.path.join(self.output_dir, "hausdorff_test_embeddings.csv"),
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
        np.savez_compressed(
            os.path.join(self.output_dir, "all_heldout_feature_embeddings.npz"), **data
        )
        print(
            f"Hausdorff {record['true_model_a']} vs {record['true_model_b']}: "
            f"{stats['hausdorff']:.6g} "
            f"(p95={stats['hausdorff_p95']:.6g}, "
            f"unit={normalized_stats['hausdorff']:.6g}, N={len(a)}+{len(b)})"
        )
        return {
            "hausdorff": stats["hausdorff"],
            "hausdorff_p95": stats["hausdorff_p95"],
            "hausdorff_unit": normalized_stats["hausdorff"],
        }

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
        embedding_metrics = self.plot_latent_space()
        hausdorff_metrics = self.evaluate_hausdorff_test_embeddings()
        future_mse = self.evaluate_heldout_future_mse()
        self.plot_prediction_examples()
        print(f"Saved results to: {os.path.abspath(self.output_dir)}")
        return {
            **embedding_metrics,
            **hausdorff_metrics,
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
                d_x=5, d_e=20, d_s=d_s, d_p=d_p,
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
    with open(os.path.join(OUTPUT_ROOT, "true_label_only_comparison.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\n=== TRUE-LABEL-ONLY EXPERIMENT COMPARISON ===")
    for result in results:
        print(result)


if __name__ == "__main__":
    run_experiment()