import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from einn.evaluation.generalization_evaluator import GeneralizationEvaluation


class EINNExperimentPlotter:
    """Create the training and generalization plots in separate methods."""

    STATE_LABELS = {
        3: ["Susceptible (S)", "Infected (I)", "Recovered (R)"],
        5: ["Susceptible (S)", "Exposed (E)", "Infected (I)", "Recovered (R)", "Mortality (M)"],
    }

    def __init__(self, output_dir: str | Path, dpi: int = 300) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_loss_curves(self, metrics: Sequence, filename: str = "loss_curves.png") -> Path | None:
        """Plot the Phase-4 loss components, following TrainingPostProcessor."""
        if not metrics:
            return None

        phase4 = [metric for metric in metrics if metric.phase == 4]
        if not phase4:
            return None

        epochs = [metric.epoch for metric in phase4]
        total_loss = [metric.total_loss for metric in phase4]
        data_t = [metric.loss_components.get("data_T", 0.0) for metric in phase4]
        data_f = [metric.loss_components.get("data_F", 0.0) for metric in phase4]
        ode_t = [metric.loss_components.get("ode_T", 0.0) for metric in phase4]
        aux = [metric.loss_components.get("aux", 0.0) for metric in phase4]
        ic = [metric.loss_components.get("ic", 0.0) for metric in phase4]

        plt.figure(figsize=(14, 8))
        plt.plot(epochs, total_loss, label="Total Loss (Phase 4)", color="black", linewidth=2.5)
        plt.plot(epochs, data_t, label="Data Loss (Time)", color="blue", linestyle="--", alpha=0.7)
        plt.plot(epochs, data_f, label="Data Loss (Feature)", color="cyan", linestyle="--", alpha=0.7)
        plt.plot(epochs, ode_t, label="ODE Physics Loss (Time)", color="red", linestyle=":", alpha=0.7)
        plt.plot(epochs, aux, label="Auxiliary Loss", color="green", linestyle="-", alpha=0.6)
        plt.plot(epochs, ic, label="IC Loss", color="purple", linestyle="-.", alpha=0.6)
        plt.yscale("log")
        plt.xlabel("Makro-Epoch")
        plt.ylabel("Veszteség (Loss Value) - Log Skála")
        plt.title("EINN Tanítási Veszteséggörbék Kialakulása")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    def plot_true_vs_predicted(
        self,
        true_states: np.ndarray,
        predicted_states: np.ndarray,
        filename: str = "true_vs_predicted.png",
    ) -> Path | None:
        """Plot all physical compartments for the full-history prediction."""
        if true_states.shape != predicted_states.shape:
            raise ValueError(
                "True and predicted state matrices must have identical shapes; "
                f"received {true_states.shape} and {predicted_states.shape}."
            )

        d_s = true_states.shape[1]
        labels = self.STATE_LABELS.get(
            d_s,
            [f"State {index}" for index in range(d_s)],
        )
        fig, axes = plt.subplots(
            nrows=d_s,
            ncols=1,
            figsize=(10, 2.5 * d_s),
            sharex=True,
            squeeze=False,
        )

        for index in range(d_s):
            axis = axes[index, 0]
            axis.plot(
                true_states[:, index],
                label=f"True {labels[index]}",
                color="black",
                linewidth=2,
            )
            axis.plot(
                predicted_states[:, index],
                label=f"Predicted {labels[index]}",
                color="red",
                linestyle="--",
                linewidth=2,
            )
            axis.set_ylabel("Populáció aránya")
            axis.grid(True, linestyle=":", alpha=0.6)
            axis.legend(loc="upper right")

        axes[-1, 0].set_xlabel("Napok (időlépések)")
        fig.suptitle(
            "Valós (Ground Truth) vs. Becsült (Predicted) Látens Állapotok",
            fontsize=14,
        )
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    @staticmethod
    def _mark_temporal_landmarks(axis, coordinates: np.ndarray, observed_steps: int) -> None:
        axis.scatter(
            coordinates[0, 0], coordinates[0, 1],
            color="red", marker="s", s=80, label="Start (0. nap)", zorder=5,
        )
        axis.scatter(
            coordinates[-1, 0], coordinates[-1, 1],
            color="black", marker="X", s=100, label="End", zorder=5,
        )
        current_idx = max(0, min(observed_steps - 1, len(coordinates) - 1))
        axis.scatter(
            coordinates[current_idx, 0], coordinates[current_idx, 1],
            color="blue", marker="o", s=80, label="Jelen", zorder=6,
        )

    def plot_generalization_grid(
        self,
        evaluation: GeneralizationEvaluation,
        d_e: int,
        filename: str = "generalization_full_analysis_target_2.png",
    ) -> Path:
        """Create the original 4x6 prediction and latent-space diagnostic grid."""
        if len(evaluation.phases) != 4:
            raise ValueError(f"Expected four phase evaluations, got {len(evaluation.phases)}.")

        fig = plt.figure(figsize=(32, 20))
        fig.suptitle(
            "EINN Generalizáció: Inferencia "
            f"({evaluation.target_name}) és Látens Tér Evolúció",
            fontsize=24,
            y=0.98,
        )

        time_index = np.arange(len(evaluation.true_target))
        for row, phase in enumerate(evaluation.phases):
            ax_pred = fig.add_subplot(4, 6, row * 6 + 1)
            ax_tsne = fig.add_subplot(4, 6, row * 6 + 2)
            ax_pca_var = fig.add_subplot(4, 6, row * 6 + 3)
            ax_pca_2d = fig.add_subplot(4, 6, row * 6 + 4)
            ax_pca_3d = fig.add_subplot(4, 6, row * 6 + 5, projection="3d")
            ax_cos = fig.add_subplot(4, 6, row * 6 + 6)

            # 1. Prediction curve.
            ax_pred.plot(
                evaluation.true_target,
                label=f"Valós {evaluation.target_name}",
                color="black",
                linewidth=2,
            )
            ax_pred.plot(
                phase.predicted_states[:, evaluation.target_idx],
                label="Prediktált",
                color="red",
                linestyle="--",
                linewidth=2,
            )
            ax_pred.axvline(
                x=phase.observed_steps,
                color="blue",
                linestyle=":",
                linewidth=1.5,
                label="Ablak vége",
            )
            ax_pred.axvspan(0, phase.observed_steps, color="blue", alpha=0.05)
            ax_pred.set_title(f"{phase.name}\nFuture MSE: {phase.mse:.4g}")
            ax_pred.grid(True, linestyle="--", alpha=0.5)
            if row == 0:
                ax_pred.legend()

            # 2. t-SNE.
            tsne = phase.tsne_coordinates
            ax_tsne.scatter(
                tsne[:, 0], tsne[:, 1], c=time_index, cmap="viridis",
                s=20, edgecolor="k", alpha=0.7,
            )
            ax_tsne.plot(tsne[:, 0], tsne[:, 1], color="gray", alpha=0.3, linewidth=1)
            self._mark_temporal_landmarks(ax_tsne, tsne, phase.observed_steps)
            ax_tsne.set_title(f"t-SNE ({d_e}D → 2D)")
            ax_tsne.grid(True, linestyle="--", alpha=0.5)
            if row == 0:
                ax_tsne.legend(fontsize="small")

            # 3. PCA explained variance.
            cumulative = phase.pca_cumulative_variance
            ax_pca_var.plot(
                np.arange(1, len(cumulative) + 1),
                cumulative,
                marker="o",
                linestyle="-",
                color="teal",
            )
            ax_pca_var.axhline(y=0.95, color="r", linestyle="--", label="95%")
            ax_pca_var.axhline(y=0.99, color="g", linestyle="--", label="99%")
            ax_pca_var.set_ylim(0, 1.02)
            ax_pca_var.set_title(f"PCA variancia (99%-hoz: {phase.n_components_99} dim)")
            ax_pca_var.set_xlabel("Főkomponensek")
            ax_pca_var.grid(True, linestyle="--", alpha=0.6)
            if row == 0:
                ax_pca_var.legend(fontsize="small")

            # 4. PCA 2D.
            pca_coords = phase.pca_coordinates
            ax_pca_2d.scatter(
                pca_coords[:, 0], pca_coords[:, 1], c=time_index,
                cmap="viridis", s=20, edgecolor="k", alpha=0.7,
            )
            ax_pca_2d.plot(
                pca_coords[:, 0], pca_coords[:, 1], color="gray", alpha=0.3, linewidth=1,
            )
            self._mark_temporal_landmarks(ax_pca_2d, pca_coords[:, :2], phase.observed_steps)
            ax_pca_2d.set_title("PCA látens tér (2D)")
            ax_pca_2d.grid(True, linestyle="--", alpha=0.5)

            # 5. PCA 3D.
            ax_pca_3d.scatter(
                pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                c=time_index, cmap="viridis", s=20, edgecolor="k", alpha=0.7,
            )
            ax_pca_3d.plot(
                pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                color="gray", alpha=0.3, linewidth=1,
            )
            current_idx = max(0, min(phase.observed_steps - 1, len(pca_coords) - 1))
            ax_pca_3d.scatter(*pca_coords[0, :3], color="red", marker="s", s=80, zorder=5)
            ax_pca_3d.scatter(*pca_coords[-1, :3], color="black", marker="X", s=100, zorder=5)
            ax_pca_3d.scatter(*pca_coords[current_idx, :3], color="blue", marker="o", s=80, zorder=6)
            ax_pca_3d.set_title("PCA látens tér (3D)")
            ax_pca_3d.xaxis.pane.fill = False
            ax_pca_3d.yaxis.pane.fill = False
            ax_pca_3d.zaxis.pane.fill = False

            # 6. Paired cosine similarity and Procrustes disparity.
            ax_cos.plot(phase.cosine_similarity, color="purple", linewidth=2)
            ax_cos.set_title(
                f"Time vs Feature (Procrustes: {phase.procrustes_disparity:.4f})"
            )
            ax_cos.set_xlabel("t")
            ax_cos.set_ylim([-1.1, 1.1])
            ax_cos.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    def save_evaluation(self, evaluation: GeneralizationEvaluation) -> Path:
        """Persist numerical diagnostics and latent arrays for later analysis."""
        metrics = {
            "target_idx": evaluation.target_idx,
            "target_name": evaluation.target_name,
            "phases": [
                {
                    "name": phase.name,
                    "observed_steps": phase.observed_steps,
                    "mse": phase.mse,
                    "procrustes_disparity": phase.procrustes_disparity,
                    "n_components_99": phase.n_components_99,
                }
                for phase in evaluation.phases
            ],
        }
        metrics_path = self.output_dir / "generalization_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)

        arrays: dict[str, np.ndarray] = {
            "true_states": evaluation.true_states,
            "true_target": evaluation.true_target,
            "full_prediction": evaluation.full_prediction,
        }
        for index, phase in enumerate(evaluation.phases):
            prefix = f"phase_{index + 1}"
            arrays[f"{prefix}_predicted_states"] = phase.predicted_states
            arrays[f"{prefix}_time_embeddings"] = phase.time_embeddings
            arrays[f"{prefix}_feature_embeddings"] = phase.feature_embeddings
            arrays[f"{prefix}_tsne"] = phase.tsne_coordinates
            arrays[f"{prefix}_pca"] = phase.pca_coordinates
            arrays[f"{prefix}_cosine_similarity"] = phase.cosine_similarity

        np.savez_compressed(self.output_dir / "generalization_latents.npz", **arrays)
        return metrics_path
