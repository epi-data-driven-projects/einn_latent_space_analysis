from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from sklearn.decomposition import PCA

# Retain the original upper bound for displayed PCA components.
PCA_COMPONENTS_TO_SHOW = 8


class SingleGeneralizationPlotter:
    """Render the original experiment reports without changing numerical results."""

    def configure_plot_style(self) -> None:
        """Apply the established publication typography, color palette, tick styles, and vector-compatible font settings.


        :return None: No return value; state or serialized artifacts are updated in place.
        """
        plt.rcParams.update({
            "font.family": "DejaVu Sans",  # DejaVu Sans supports the accented Hungarian figure labels.
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
        })

    def _panel_style(self, ax, time_axis: bool = False):
        """Apply a consistent readable visual style to one Matplotlib axes.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param bool time_axis: Whether the panel should use the common simulation time-axis formatting.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.65)
        if time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    def _mark_observed(self, ax, end: int, horizon: int):
        # Highlight the time range actually supplied to the feature encoder.
        """Shade the observed prefix and draw its cutoff in a forecast panel.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param int end: Last observed index or endpoint of the selected interval.
        :param int horizon: Total number of simulation time points in the trajectory.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        ax.axvspan(0, end - 1, color="#EDF3FA", alpha=0.65, zorder=0)
        ax.axvline(end - 0.5, color="#8B99AC", linestyle="--", linewidth=1.1, zorder=2)
        ax.set_xlim(0, horizon - 1)

    def plot_results(self,
                     phase_results: list[dict],
                     shared_pca: PCA,
                     true_target: np.ndarray,
                     target_name: str,
                     output_path: Path,
                     model_name: str,
                     embedding_dim: int,
                     ) -> None:
        """Render the original multi-column single-model generalization diagnostic figure.


        :param list[dict] phase_results: Predictions and embedding arrays for every selected observation window.
        :param PCA shared_pca: PCA estimator already fitted to the combined latent embeddings.
        :param np.ndarray true_target: Noise-free target-compartment reference trajectory.
        :param str target_name: Human-readable name of the epidemiological target compartment.
        :param Path output_path: Destination path for the rendered figure.
        :param str model_name: Name of the mechanistic epidemic model, such as SIR, SIS, or ROST.
        :param int embedding_dim: Dimensionality of each model-generated latent representation.
        :return None: No return value; state or serialized artifacts are updated in place.
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

        # Prepare figure-wide legends once, outside individual panels.
        prediction_handles = None
        pca_handles = None

        for row, result in enumerate(phase_results):
            end = result["t_end"]
            title = f"Megfigyelt: {end} időpont"

            # ==========================================================
            # Show the predicted epidemiological target against the clean reference.
            # ==========================================================
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

            # Report the future-only reconstruction error within each prediction panel.
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

            # ==========================================================
            # Display cumulative PCA explained variance for the current window.
            # ==========================================================
            ax = axes[row, 1]
            cum = result["cum_variance"]

            # Limit visible bars without changing the complete PCA threshold calculation.
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

            # ==========================================================
            # 3. PCA 2D
            # ==========================================================
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

            # ==========================================================
            # Compare Time and Feature embeddings with cosine similarity across the horizon.
            # ==========================================================
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

            # ==========================================================
            # 5. PROCRUSTES
            # ==========================================================
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

        # ==============================================================
        # Place shared legends above the subplots to preserve readable plot areas.
        # ==============================================================

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
