from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
import numpy as np

# Visual constants retained from the original experiment.
PALETTE = {"ink": "#18283D", "muted": "#607185", "grid": "#D8E1EB",
           "teal": "#008E97", "teal_dark": "#006A76", "blue": "#416BA8",
           "navy": "#25415D", "orange": "#D78A30", "magenta": "#AB4C77",
           "gray": "#8192A6", "background_obs": "#E5F0F4"}



class DualInferencePlotter:
    """Render the original four-by-four SIR dual-inference comparison."""

    def configure_plot_style(self) -> None:
        """Apply the established publication typography, color palette, tick styles, and vector-compatible font settings.

        :return None: None; this operation updates existing state.
        """
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.titlesize": 20,
            "axes.titleweight": "semibold",
            "axes.labelsize": 18,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "#AAB8C8",
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.8,
            "grid.linewidth": 0.75,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        })

    def _style_panel(self, ax, time_axis: bool = False) -> None:
        """Style panel while preserving the original computational behavior.

        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param bool time_axis: Whether the panel should use the common simulation time-axis formatting.
        :return None: None; this operation updates existing state.
        """
        ax.set_axisbelow(True)
        ax.grid(True, which="major")
        ax.tick_params(axis="both", labelsize=16, length=4)
        if time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    def _finite_limits(self, values, relative_pad: float = 0.10, min_span: float = 0.01):
        """Determine common finite axis bounds with stable padding across compared trajectories and observation windows.

        :param Any values: Numerical values to summarize, normalize, or visualize.
        :param float relative_pad: Relative padding added around the observed numerical range.
        :param float min_span: Lower bound on plot range to avoid degenerate axes.
        :return Any: The result of this operation.
        """
        data = np.asarray(values, dtype=float).reshape(-1)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return -1.0, 1.0
        lo, hi = float(data.min()), float(data.max())
        span = max(hi - lo, min_span)
        return lo - span * relative_pad, hi + span * relative_pad

    def _plot_pca_pair(self, ax, coords_a, coords_b, t_end: int, title: str,
                       xlim, ylim):
        """Draw two time-resolved latent trajectories in the same two-dimensional PCA coordinate system.

        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param Any coords_a: Two-dimensional PCA coordinates of the first compared trajectory.
        :param Any coords_b: Two-dimensional PCA coordinates of the second compared trajectory.
        :param int t_end: Number of historical time coordinates exposed to the Feature module.
        :param str title: Descriptive title shown above the plot or comparison.
        :param Any xlim: Horizontal axis range shared by the relevant plot panels.
        :param Any ylim: Vertical axis range shared by the relevant plot panels.
        :return Any: The result of this operation.
        """
        for coords, color in (
            (coords_a, PALETTE["teal"]),
            (coords_b, PALETTE["blue"]),
        ):
            coords = np.asarray(coords, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(f"Hibás PCA-koordináták: {coords.shape}")
            if not 1 <= t_end <= len(coords):
                raise ValueError(f"Hibás megfigyelési ablak: {t_end}")

            ax.plot(coords[:, 0], coords[:, 1], color=color,
                    lw=2.45, alpha=.88, solid_capstyle="round", zorder=2)
            ax.scatter(coords[:, 0], coords[:, 1], s=10,
                       color=color, alpha=.42, edgecolors="none", zorder=3)
            # Use the same start/end/observation markers in every projected trajectory.
            ax.scatter(*coords[0], marker="x", color=PALETTE["ink"],
                       s=60, linewidths=1.75, zorder=5)
            ax.scatter(*coords[-1], marker="s", color=PALETTE["orange"],
                       s=61, zorder=6)
            ax.scatter(*coords[t_end-1], marker="o", facecolors="none",
                       edgecolors=PALETTE["magenta"], s=120,
                       linewidths=2.1, zorder=7)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=9)
        ax.set_xlabel("1. főkomponens")
        ax.set_ylabel("2. főkomponens")
        self._style_panel(ax)

    def plot_dual_inference_results(self, phase_results, true_target_1, true_target_2,
                                    output_dir, embedding_dim: int):
        """Render the original 4x4 comparison of epidemiological forecasts and three PCA bases for two SIR parameterizations.

        :param Any phase_results: Evaluation measurements grouped by observation-window length.
        :param Any true_target_1: Noise-free target trajectory of the first mechanistic parameterization.
        :param Any true_target_2: Noise-free target trajectory of the second mechanistic parameterization.
        :param Any output_dir: Directory where generated results are written.
        :param int embedding_dim: Dimensionality of each model-generated latent representation.
        :return Any: The result of this operation.
        """
        if len(phase_results) != 4:
            raise ValueError("A 4×4 elrendezéshez pontosan négy megfigyelési ablak kell.")

        self.configure_plot_style()
        target_a = np.asarray(true_target_1, dtype=float).reshape(-1)
        target_b = np.asarray(true_target_2, dtype=float).reshape(-1)
        if target_a.shape != target_b.shape:
            raise ValueError("A két referencia-idősor hossza nem egyezik.")
        horizon = len(target_a)
        times = np.arange(horizon)

        pca_keys = (
            ("pca_common_1", "pca_common_2", "Közös PCA-vetítés"),
            ("pca_a_1", "pca_a_2", "A trajektória PCA-bázisa"),
            ("pca_b_1", "pca_b_2", "B trajektória PCA-bázisa"),
        )
        # Apply consistent axes across all rows within each distinct PCA basis.
        # Different PCA bases may require different coordinate ranges.
        pca_limits = []
        for a_key, b_key, _ in pca_keys:
            combined = np.concatenate([
                np.asarray(phase[key], dtype=float)
                for phase in phase_results
                for key in (a_key, b_key)
            ], axis=0)
            pca_limits.append((
                self._finite_limits(combined[:, 0], min_span=.02),
                self._finite_limits(combined[:, 1], min_span=.02),
            ))

        predictions = [
            np.asarray(phase[f"results_{i}"]["predicted_trajectory"], dtype=float).reshape(-1)
            for phase in phase_results for i in (1, 2)
        ]
        if any(p.size != horizon for p in predictions):
            raise ValueError("Az egyik becsült trajektória hossza eltér a referenciától.")
        pred_limits = self._finite_limits(
            np.concatenate([target_a, target_b, *predictions]),
            relative_pad=.08, min_span=.015,
        )

        fig = plt.figure(figsize=(29.5, 25.0), facecolor="white")
        grid = fig.add_gridspec(
            4, 4, left=.058, right=.986, top=.845, bottom=.055,
            wspace=.52, hspace=.68,
        )
        fig.suptitle(
            "EINN · két eltérő SIR-járványdinamika összehasonlítása"
            f"   |   $d_e={embedding_dim}$",
            fontsize=29, fontweight="bold", color=PALETTE["ink"], y=.986,
        )
        fig.text(
            .5, .952,
            "Négy megfigyelési ablak · két független teszttrajektória · összehasonlítható PCA-tengelyek",
            ha="center", fontsize=20, color=PALETTE["muted"],
        )

        prediction_legend = [
            Line2D([0], [0], color=PALETTE["teal_dark"], lw=2.4,
                   label=r"A referencia ($\beta=0.99$, $\gamma=0.1$)"),
            Line2D([0], [0], color=PALETTE["teal"], lw=2.7,
                   ls=(0, (5, 2)), label="A becslés"),
            Line2D([0], [0], color=PALETTE["navy"], lw=2.4,
                   label=r"B referencia ($\beta=0.01$, $\gamma=0.1$)"),
            Line2D([0], [0], color=PALETTE["blue"], lw=2.7,
                   ls=(0, (5, 2)), label="B becslés"),
            Patch(facecolor=PALETTE["background_obs"], label="Megfigyelt időszak"),
        ]
        fig.legend(handles=prediction_legend, loc="upper center",
                   bbox_to_anchor=(.5, .928), ncol=5, frameon=False, fontsize=18)
        marker_legend = [
            Line2D([0], [0], color=PALETTE["ink"], marker="x", lw=0,
                   ms=8, label="Látens pálya kezdete"),
            Line2D([0], [0], color=PALETTE["orange"], marker="s", lw=0,
                   ms=7, label="Látens pálya végpontja"),
            Line2D([0], [0], color=PALETTE["magenta"], marker="o", lw=0,
                   markerfacecolor="none", ms=9, markeredgewidth=1.8,
                   label="Megfigyelés vége"),
        ]
        fig.legend(handles=marker_legend, loc="upper center",
                   bbox_to_anchor=(.5, .897), ncol=3, frameon=False, fontsize=18)

        for row, phase in enumerate(phase_results):
            end = int(phase["t_end"])
            if not 1 <= end < horizon:
                raise ValueError(f"Érvénytelen megfigyelési ablak: {end}")

            r_a, r_b = phase["results_1"], phase["results_2"]
            pred_a = np.asarray(r_a["predicted_trajectory"], dtype=float).reshape(-1)
            pred_b = np.asarray(r_b["predicted_trajectory"], dtype=float).reshape(-1)

            ax = fig.add_subplot(grid[row, 0])
            ax.plot(times, target_a, color=PALETTE["teal_dark"],
                    lw=2.35, zorder=3)
            ax.plot(times, pred_a, color=PALETTE["teal"],
                    lw=2.7, ls=(0, (5, 2)), zorder=4)
            ax.plot(times, target_b, color=PALETTE["navy"],
                    lw=2.35, zorder=3)
            ax.plot(times, pred_b, color=PALETTE["blue"],
                    lw=2.7, ls=(0, (5, 2)), zorder=4)
            ax.axvspan(-.5, end-.5, color=PALETTE["background_obs"],
                       alpha=.8, zorder=0)
            ax.axvline(end-.5, color=PALETTE["muted"],
                       lw=1.45, ls=(0, (4, 3)), zorder=5)
            ax.set_xlim(0, horizon-1)
            ax.set_ylim(*pred_limits)
            ax.set_title(f"{end} megfigyelt időpont · járványdinamika", pad=9)
            ax.set_xlabel("Időindex")
            ax.set_ylabel("Fertőzöttek aránya")
            ax.text(
                .97, .96,
                f"Jövőbeli MSE  A: {float(r_a['mse_future']):.3g}\n"
                f"Jövőbeli MSE  B: {float(r_b['mse_future']):.3g}",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=15, color=PALETTE["ink"],
                bbox=dict(boxstyle="round,pad=.34", facecolor="white",
                          edgecolor=PALETTE["grid"], alpha=.93),
                zorder=10,
            )
            self._style_panel(ax, time_axis=True)

            for j, (a_key, b_key, title) in enumerate(pca_keys, start=1):
                ax = fig.add_subplot(grid[row, j])
                xlim, ylim = pca_limits[j-1]
                self._plot_pca_pair(ax, phase[a_key], phase[b_key], end,
                               title, xlim, ylim)

        filename = os.path.join(output_dir, "sir_dual_inference_pca_comparison.png")
        fig.savefig(filename, dpi=250, bbox_inches="tight", pad_inches=.18)
        pdf_filename = os.path.splitext(filename)[0] + ".pdf"
        fig.savefig(pdf_filename, bbox_inches="tight", pad_inches=.18)
        plt.close(fig)

        # Return the one original four-by-four figure without extra panel files.
        return filename, pdf_filename
