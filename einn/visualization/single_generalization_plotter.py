"""
Multi-column 4x5 generalization diagnostic plotter for single-model EINN experiments.

Renders per-window epidemiological forecasts, cumulative PCA explained variance, shared 2D
PCA latent trajectories, pointwise Time-Feature cosine similarity, and pointwise Procrustes
alignment residuals (thesis Sections 2.5 and 4.1).
"""
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Maximum number of leading principal components shown in the cumulative variance panels.
PCA_COMPONENTS_TO_SHOW: int = 8


class SingleGeneralizationPlotter:
    """
    Render the 4x5 single-model generalization diagnostic figure from evaluation results.

    Organizes four observation windows as rows across five columns: target forecast,
    cumulative PCA variance, shared 2D PCA latent trajectory, Time-Feature cosine
    similarity, and pointwise Procrustes alignment residual.
    """

    def configure_plot_style(self) -> None:
        """
        Apply publication typography, spine removal, grid styling, and TrueType font settings.

        :return None: Updates matplotlib.pyplot.rcParams in place.
        """
        plt.rcParams.update({
            "font.family": "DejaVu Sans",  # Supports accented Hungarian figure labels.
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#ABB6C5",
            "axes.labelcolor": "#27364A",
            "text.color": "#1A2940",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#DAE2EC",
            "grid.alpha": 0.75,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # Embed TrueType fonts in vector outputs.
            "ps.fonttype": 42,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
        })

    def _panel_style(self, ax: Axes, time_axis: bool = False) -> None:
        """
        Apply uniform gridlines and optional integer time-index ticks to a subplot panel.

        :param Axes ax: Target Matplotlib Axes instance.
        :param bool time_axis: If True, restricts x-axis major ticks to at most 5 integers.
        :return None: Modifies ax in place.
        """
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.65)
        if time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    def _mark_observed(self, ax: Axes, end: int, horizon: int) -> None:
        """
        Shade the observed prefix [0, end - 1] and draw a dashed vertical cutoff at end - 0.5.

        :param Axes ax: Target Matplotlib Axes instance.
        :param int end: Number of historical timesteps supplied to FeatureModule.
        :param int horizon: Total trajectory length T.
        :return None: Modifies ax in place.
        """
        ax.axvspan(0, end - 1, color="#EDF3FA", alpha=0.65, zorder=0)
        ax.axvline(end - 0.5, color="#8B99AC", linestyle="--", linewidth=1.1, zorder=2)
        ax.set_xlim(0, horizon - 1)

    def plot_results(
        self,
        phase_results: list[dict[str, Any]],
        shared_pca: PCA,
        true_target: np.ndarray,
        target_name: str,
        output_path: Path,
        model_name: str,
        embedding_dim: int,
    ) -> None:
        """
        Render and save the 4x5 single-model generalization diagnostic figure.

        Each entry in phase_results corresponds to one observation window and must provide:
        't_end' (int), 'predicted_trajectory' of shape (T,), 'mse_future' (float),
        'cum_variance' of shape (min(T, d_e),), 'pca_95' (int), 'pca_99' (int),
        'pca_coords' of shape (T, 2), 'cosine_similarity' of shape (T,),
        'procrustes_pointwise_residual' of shape (T,), and 'procrustes_disparity' (float).

        :param list[dict[str, Any]] phase_results: List of exactly 4 per-window evaluation dictionaries.
        :param PCA shared_pca: PCA estimator fitted jointly on all Feature embeddings (Section 2.5).
        :param np.ndarray true_target: Noise-free reference target trajectory of shape (T,).
        :param str target_name: Y-axis label for the target epidemiological compartment.
        :param Path output_path: Destination file path for the rendered PNG figure.
        :param str model_name: Epidemiological model identifier (e.g. SIR, SIS, ROST).
        :param int embedding_dim: Latent embedding dimension d_e displayed in the title.
        :return None: Saves the figure to output_path and closes it.
        :raises ValueError: If phase_results does not contain exactly 4 observation windows.
        """
        if len(phase_results) != 4:
            raise ValueError("Pontosan négy megfigyelési ablak szükséges.")

        plt.rcParams.update({
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        })

        fig, axes = plt.subplots(
            nrows=4,
            ncols=5,
            figsize=(30, 20),
            squeeze=False
        )

        times = np.arange(len(true_target))

        fig.suptitle(
            f"EINN általánosításvizsgálat – {model_name} | Látens dimenzió: {embedding_dim}",
            fontsize=18,
            y=0.985
        )

        # Collect legend handles once from the first row for figure-wide legends.
        prediction_handles = None
        pca_handles = None

        for row, result in enumerate(phase_results):
            end = result["t_end"]
            title = f"Megfigyelt: {end} időpont"

            # Column 0: Noise-free reference vs. Feature-branch forecast and out-of-window MSE.
            ax = axes[row, 0]

            line_true, = ax.plot(
                times,
                true_target,
                color="black",
                linewidth=1.8,
                label="Valódi járványgörbe"
            )
            line_pred, = ax.plot(
                times,
                result["predicted_trajectory"],
                color="teal",
                linewidth=1.8,
                label="Modellbecslés"
            )
            line_obs = ax.axvline(
                end,
                color="gray",
                linestyle="--",
                linewidth=1.1,
                label="Megfigyelés vége"
            )
            ax.axvspan(0, end, color="gray", alpha=0.07)

            ax.set_title("Járványgörbe és előrejelzés", fontweight="bold")
            ax.set_xlabel("Időindex")
            ax.set_ylabel(target_name)
            ax.grid(alpha=0.25)

            # Annotate observation window length and strictly future MSE (t >= end).
            info_text = (
                f"Megfigyelt: {end} időpont\n"
                f"Jövőbeli négyzetes hiba: {result['mse_future']:.3g}"
            )
            ax.text(
                0.98, 0.95,
                info_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="lightgray",
                    alpha=0.9
                )
            )

            if row == 0:
                prediction_handles = [line_true, line_pred, line_obs]

            # Column 1: Per-window cumulative PCA explained variance with 95% and 99% thresholds.
            ax = axes[row, 1]
            cum = result["cum_variance"]

            # Cap displayed components at PCA_COMPONENTS_TO_SHOW without altering pca_95/pca_99.
            max_show = min(PCA_COMPONENTS_TO_SHOW, len(cum))
            shown_x = np.arange(1, max_show + 1)
            shown_y = cum[:max_show]

            ax.plot(
                shown_x,
                shown_y,
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color="royalblue"
            )
            ax.axhline(0.95, color="darkorange", linestyle="--", linewidth=1.0)
            ax.axhline(0.99, color="mediumpurple", linestyle=":", linewidth=1.0)

            ax.set_xlim(0.8, max_show + 0.2)
            ax.set_xticks(shown_x)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Főkomponensek száma")
            ax.set_ylabel("Kumulatív varianciaarány")
            ax.set_title("Kumulatív magyarázott variancia", fontweight="bold")
            ax.grid(alpha=0.25)

            ax.text(
                0.97, 0.08,
                f"95%-hoz: {result['pca_95']} komponens\n"
                f"99%-hoz: {result['pca_99']} komponens",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="lightgray",
                    alpha=0.9
                )
            )

            # Column 2: 2D Feature latent trajectory (T, 2) projected onto shared_pca.
            ax = axes[row, 2]
            coords = result["pca_coords"]

            traj_line, = ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="gray",
                alpha=0.45,
                linewidth=1.0,
                zorder=1
            )

            pts = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=16,
                color="steelblue",
                alpha=0.95,
                edgecolors="none",
                zorder=2
            )

            # Mark trajectory start (t = 0), end (t = T - 1), and observation cutoff (t = end - 1).
            start_pt = ax.scatter(
                *coords[0],
                color="black",
                marker="x",
                s=45,
                linewidths=1.2,
                zorder=3,
                label="Kezdet"
            )

            end_pt = ax.scatter(
                *coords[-1],
                color="darkorange",
                marker="s",
                s=35,
                zorder=3,
                label="Végpont"
            )

            obs_pt = ax.scatter(
                *coords[end - 1],
                facecolors="none",
                edgecolors="crimson",
                linewidths=1.4,
                s=70,
                zorder=4,
                label="Megfigyelés vége"
            )

            ax.set_title("Látens pálya – közös PCA-vetítés", fontweight="bold")
            ax.set_xlabel("1. főkomponens")
            ax.set_ylabel("2. főkomponens")
            ax.grid(alpha=0.25)

            ax.text(
                0.03, 0.05,
                f"PC1: {shared_pca.explained_variance_ratio_[0]:.1%}\n"
                f"PC2: {shared_pca.explained_variance_ratio_[1]:.1%}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="lightgray",
                    alpha=0.9
                )
            )

            if row == 0:
                pca_handles = [start_pt, end_pt, obs_pt]

            # Column 3: Pointwise Time-Feature cosine similarity across the full horizon (T,).
            ax = axes[row, 3]
            ax.plot(
                times,
                result["cosine_similarity"],
                color="slateblue",
                linewidth=1.5
            )
            ax.axvline(end, color="gray", linestyle="--", linewidth=1.0)
            ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
            ax.axvspan(0, end, color="gray", alpha=0.07)

            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("Időindex")
            ax.set_ylabel("Koszinuszos hasonlóság")
            ax.set_title("Idő- és jellemzőmodul hasonlósága", fontweight="bold")
            ax.grid(alpha=0.25)

            # Column 4: Pointwise Procrustes alignment residual (T,) and scalar disparity D_Proc.
            ax = axes[row, 4]
            residual = result["procrustes_pointwise_residual"]

            ax.plot(
                times,
                residual,
                color="chocolate",
                linewidth=1.4
            )
            ax.axvline(end, color="gray", linestyle="--", linewidth=1.0)
            ax.axvspan(0, end, color="gray", alpha=0.07)

            ax.set_xlabel("Időindex")
            ax.set_ylabel("Illesztés utáni eltérés")
            ax.set_title("Procrustes-illesztés maradéka", fontweight="bold")
            ax.grid(alpha=0.25)

            ax.text(
                0.97, 0.95,
                f"Összesített eltérés: {result['procrustes_disparity']:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="lightgray",
                    alpha=0.9
                )
            )

        # Place shared legends above the subplot grid to avoid overlapping data curves.
        if prediction_handles is not None:
            fig.legend(
                handles=prediction_handles,
                loc="upper left",
                bbox_to_anchor=(0.03, 0.955),
                ncol=3,
                frameon=True
            )

        if pca_handles is not None:
            fig.legend(
                handles=pca_handles,
                loc="upper center",
                bbox_to_anchor=(0.57, 0.955),
                ncol=3,
                frameon=True
            )

        fig.tight_layout(rect=[0.02, 0.03, 0.995, 0.95], w_pad=2.2, h_pad=2.2)
        fig.savefig(output_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
