"""
Publication-quality 4x4 plotter for comparing two SIR parameterizations under one trained EINN.

Generates the dual-inference forecast and three-basis 2D PCA comparison figure analyzed in
thesis Section 4.1 (Figure 15), exporting both raster PNG and vector PDF artifacts.
"""
import os
from typing import Any, Dict, List, Sequence, Tuple

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

# Shared publication color palette for dual-inference trajectories, PCA curves, and markers.
PALETTE: Dict[str, str] = {
    "ink": "#18283D", "muted": "#607185", "grid": "#D8E1EB",
    "teal": "#008E97", "teal_dark": "#006A76", "blue": "#416BA8",
    "navy": "#25415D", "orange": "#D78A30", "magenta": "#AB4C77",
    "gray": "#8192A6", "background_obs": "#E5F0F4",
}


class DualInferencePlotter:
    """
    Render the 4x4 dual-inference SIR comparison figure (thesis Section 4.1, Figure 15).

    Displays four observation windows as rows across four columns: (1) target forecasts vs.
    noise-free references for two SIR parameterizations A and B, (2) shared PCA projection,
    (3) trajectory A PCA basis, and (4) trajectory B PCA basis.
    """

    def configure_plot_style(self) -> None:
        """
        Apply publication typography, spine removal, grid styling, and TrueType PDF font settings.

        :return None: Updates matplotlib.pyplot.rcParams in place.
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
            "pdf.fonttype": 42,  # Embed TrueType fonts for editable vector PDFs.
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        })

    def _style_panel(self, ax: Axes, time_axis: bool = False) -> None:
        """
        Apply uniform grid and tick formatting to a single subplot panel.

        :param Axes ax: Matplotlib Axes instance to format.
        :param bool time_axis: If True, forces integer x-axis ticks for discrete time indices.
        :return None: Modifies ax in place.
        """
        ax.set_axisbelow(True)
        ax.grid(True, which="major")
        ax.tick_params(axis="both", labelsize=16, length=4)
        if time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    def _finite_limits(
        self,
        values: np.ndarray | Sequence[float],
        relative_pad: float = 0.10,
        min_span: float = 0.01,
    ) -> Tuple[float, float]:
        """
        Compute padded (low, high) axis limits over all finite entries in values.

        :param np.ndarray | Sequence[float] values: Array or sequence of numeric values of any shape.
        :param float relative_pad: Fractional margin added to both ends of the data span.
        :param float min_span: Minimum span enforced to prevent degenerate zero-width axes.
        :return Tuple[float, float]: Padded axis bounds (low, high); returns (-1.0, 1.0) if no finite values exist.
        """
        data = np.asarray(values, dtype=float).reshape(-1)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return -1.0, 1.0
        lo, hi = float(data.min()), float(data.max())
        span = max(hi - lo, min_span)
        return lo - span * relative_pad, hi + span * relative_pad

    def _plot_pca_pair(
        self,
        ax: Axes,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
        t_end: int,
        title: str,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
    ) -> None:
        """
        Plot two time-ordered 2D PCA latent trajectories in a shared coordinate frame.

        Marks the trajectory start (t = 0), end (t = T - 1), and observation cutoff (t = t_end - 1)
        with consistent symbols across both parameterizations.

        :param Axes ax: Target Matplotlib Axes panel.
        :param np.ndarray coords_a: 2D PCA coordinates for trajectory A of shape (T, 2).
        :param np.ndarray coords_b: 2D PCA coordinates for trajectory B of shape (T, 2).
        :param int t_end: Observation window length in [1, T].
        :param str title: Panel title identifying the PCA projection basis.
        :param Tuple[float, float] xlim: Shared horizontal axis bounds (xmin, xmax).
        :param Tuple[float, float] ylim: Shared vertical axis bounds (ymin, ymax).
        :return None: Draws trajectories and markers onto ax in place.
        :raises ValueError: If either coordinate array is not of shape (T, 2) or t_end is out of bounds.
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

            # Draw the continuous latent path and per-timestep scatter points.
            ax.plot(coords[:, 0], coords[:, 1], color=color,
                    lw=2.45, alpha=.88, solid_capstyle="round", zorder=2)
            ax.scatter(coords[:, 0], coords[:, 1], s=10,
                       color=color, alpha=.42, edgecolors="none", zorder=3)

            # Highlight start (x), end (square), and observation-window boundary (hollow circle).
            ax.scatter(*coords[0], marker="x", color=PALETTE["ink"],
                       s=60, linewidths=1.75, zorder=5)
            ax.scatter(*coords[-1], marker="s", color=PALETTE["orange"],
                       s=61, zorder=6)
            ax.scatter(*coords[t_end - 1], marker="o", facecolors="none",
                       edgecolors=PALETTE["magenta"], s=120,
                       linewidths=2.1, zorder=7)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=9)
        ax.set_xlabel("1. főkomponens")
        ax.set_ylabel("2. főkomponens")
        self._style_panel(ax)

    def plot_dual_inference_results(
        self,
        phase_results: List[Dict[str, Any]],
        true_target_1: np.ndarray,
        true_target_2: np.ndarray,
        output_dir: str | os.PathLike[str],
        embedding_dim: int,
    ) -> Tuple[str, str]:
        """
        Render and save the 4x4 dual-inference SIR comparison figure as PNG and PDF.

        Each entry in phase_results corresponds to one observation window and must provide
        't_end', 'results_1', 'results_2' (each containing 'predicted_trajectory' of shape (T,)
        and scalar 'mse_future'), and 2D PCA arrays of shape (T, 2) for keys 'pca_common_1/2',
        'pca_a_1/2', and 'pca_b_1/2'.

        :param List[Dict[str, Any]] phase_results: List of exactly 4 per-window evaluation dictionaries.
        :param np.ndarray true_target_1: Noise-free reference target for trajectory A of shape (T,).
        :param np.ndarray true_target_2: Noise-free reference target for trajectory B of shape (T,).
        :param str | os.PathLike[str] output_dir: Output directory where PNG and PDF files are written.
        :param int embedding_dim: Latent embedding dimension d_e displayed in the figure title.
        :return Tuple[str, str]: Pair of saved file paths (png_path, pdf_path).
        :raises ValueError: If phase_results does not have length 4, trajectory lengths mismatch,
            or any observation window t_end is outside [1, T - 1].
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

        # Three PCA columns: joint basis, trajectory-A basis, and trajectory-B basis (Section 4.1).
        pca_keys = (
            ("pca_common_1", "pca_common_2", "Közös PCA-vetítés"),
            ("pca_a_1", "pca_a_2", "A trajektória PCA-bázisa"),
            ("pca_b_1", "pca_b_2", "B trajektória PCA-bázisa"),
        )

        # Compute shared (xlim, ylim) per PCA column across all 4 rows for direct visual comparison.
        pca_limits: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
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

        # Compute shared y-axis limits across all forecast panels and both reference curves.
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

        # Top figure legends: trajectory line styles and latent-space markers.
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

        # Populate each row: forecast panel (col 0) followed by the three PCA basis panels (cols 1–3).
        for row, phase in enumerate(phase_results):
            end = int(phase["t_end"])
            if not 1 <= end < horizon:
                raise ValueError(f"Érvénytelen megfigyelési ablak: {end}")

            r_a, r_b = phase["results_1"], phase["results_2"]
            pred_a = np.asarray(r_a["predicted_trajectory"], dtype=float).reshape(-1)
            pred_b = np.asarray(r_b["predicted_trajectory"], dtype=float).reshape(-1)

            # Column 0: Ground-truth vs. predicted target trajectories with shaded observation window.
            ax = fig.add_subplot(grid[row, 0])
            ax.plot(times, target_a, color=PALETTE["teal_dark"],
                    lw=2.35, zorder=3)
            ax.plot(times, pred_a, color=PALETTE["teal"],
                    lw=2.7, ls=(0, (5, 2)), zorder=4)
            ax.plot(times, target_b, color=PALETTE["navy"],
                    lw=2.35, zorder=3)
            ax.plot(times, pred_b, color=PALETTE["blue"],
                    lw=2.7, ls=(0, (5, 2)), zorder=4)
            ax.axvspan(-.5, end - .5, color=PALETTE["background_obs"],
                       alpha=.8, zorder=0)
            ax.axvline(end - .5, color=PALETTE["muted"],
                       lw=1.45, ls=(0, (4, 3)), zorder=5)
            ax.set_xlim(0, horizon - 1)
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

            # Columns 1–3: 2D PCA trajectories under the common, A-fitted, and B-fitted bases.
            for j, (a_key, b_key, title) in enumerate(pca_keys, start=1):
                ax = fig.add_subplot(grid[row, j])
                xlim, ylim = pca_limits[j - 1]
                self._plot_pca_pair(ax, phase[a_key], phase[b_key], end,
                                    title, xlim, ylim)

        # Export raster PNG (250 DPI) and publication-ready vector PDF.
        filename = os.path.join(output_dir, "sir_dual_inference_pca_comparison.png")
        fig.savefig(filename, dpi=250, bbox_inches="tight", pad_inches=.18)
        pdf_filename = os.path.splitext(filename)[0] + ".pdf"
        fig.savefig(pdf_filename, bbox_inches="tight", pad_inches=.18)
        plt.close(fig)

        return filename, pdf_filename
