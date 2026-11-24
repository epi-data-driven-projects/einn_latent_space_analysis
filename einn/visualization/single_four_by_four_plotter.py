"""
Publication-quality 4x4+1 plotter and CSV exporter for persisted single-model EINN experiments.

Renders the multi-window epidemiological forecast and latent-space diagnostic figures
analyzed in thesis Section 4.1 (Figures 11-14) and Appendix (Figures 20-21), including
shared 2D PCA trajectories, per-window cumulative explained variance, pointwise Time-Feature
cosine similarity, and full-dimensional Procrustes disparity bars.
"""
from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from einn.model.interface.saved_experiment import Experiment

# Default directory layout and sweep settings for single-model generalization reports.
DEFAULT_ROOT: Path = Path("./experiment_results/generalization")
RUN_ALL_MODELS: bool = True
ALL_MODEL_NAMES: tuple[str, ...] = ("SIR", "SIS", "ROST")
ALL_EMBEDDING_DIMS: tuple[int, ...] = (20, 5)
OUTPUT_SUBDIRECTORY: str = "ujrarajzolt_abrak"

# Default diagnostic and figure-export settings (Section 4.1 uses windows 10, 18, 40, 80).
DEFAULT_PCA_COMPONENTS: int = 8
DEFAULT_OBSERVATION_WINDOWS: tuple[int, ...] = (10, 18, 40, 80)
PNG_DPI: int = 230

# Shared report color palette for forecasts, PCA curves, and cosine-similarity panels.
PALETTE: dict[str, str] = {
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

# Distinct bar colors for the 4 observation windows in the Procrustes disparity panel.
PROCRUSTES_COLORS: tuple[str, ...] = (
    "#3B7A57",  # Forest green, visually distinct from the main report curves.
    "#6B4E9B",  # Deep purple, visually distinct from the main report curves.
    "#C44536",  # Muted vermilion, visually distinct from the main report curves.
    "#8C6D46",  # Warm bronze, visually distinct from the main report curves.
)


class SingleModelFourByFourPlotter:
    """
    Render single-model EINN diagnostic figures and numerical summaries from an Experiment object.

    Produces the main 4x4+1 summary grid (forecast, cumulative PCA variance, shared 2D PCA
    trajectory, pointwise cosine similarity, and bottom Procrustes bar chart), standalone
    detail plots for cosine similarity and Procrustes disparity, and a summary CSV table.
    """

    def style(self) -> None:
        """
        Apply publication typography, spine removal, grid styling, and TrueType PDF font settings.

        :return None: Updates matplotlib.pyplot.rcParams in place.
        """
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.titlesize": 20.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 18.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "#AAB8C8",
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.8,
            "grid.linewidth": 0.9,
            "pdf.fonttype": 42,  # Embed TrueType fonts for editable vector PDFs.
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })

    def _finite_limits(
        self,
        values: np.ndarray,
        *,
        floor: float | None = None,
        ceiling: float | None = None,
        relative_pad: float = 0.12,
        min_span: float = 0.015,
    ) -> tuple[float, float]:
        """
        Compute padded (low, high) axis limits over all finite entries with optional clamping.

        :param np.ndarray values: Array of numeric values of arbitrary shape.
        :param float | None floor: Optional hard lower bound for the returned axis minimum.
        :param float | None ceiling: Optional hard upper bound for the returned axis maximum.
        :param float relative_pad: Fractional margin added to both ends of the observed span.
        :param float min_span: Minimum span enforced to avoid degenerate zero-width axes.
        :return tuple[float, float]: Padded and clamped axis limits (low, high).
        """
        vals = np.asarray(values, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (-0.05, 1.05) if floor is None else (floor, ceiling or floor + 1)

        low, high = float(vals.min()), float(vals.max())
        span = max(high - low, min_span)
        pad = max(span * relative_pad, min_span * .30)
        low, high = low - pad, high + pad

        if floor is not None:
            low = max(floor, low)
        if ceiling is not None:
            high = min(ceiling, high)

        # Fallback guard if aggressive clamping collapses the interval.
        if high <= low:
            middle = float(np.mean(vals))
            low = max(floor, middle - .02) if floor is not None else middle - .02
            high = min(ceiling, middle + .02) if ceiling is not None else middle + .02
        return low, high

    def _mark_observed(self, ax: Axes, n: int, horizon: int) -> None:
        """
        Shade the observed prefix [0, n - 1] and draw a dashed vertical cutoff at n - 0.5.

        :param Axes ax: Target Matplotlib Axes panel.
        :param int n: Number of observed timesteps exposed to FeatureModule.
        :param int horizon: Total trajectory length T.
        :return None: Modifies ax in place.
        """
        ax.axvspan(-0.5, n - 0.5, color=PALETTE["background_obs"], alpha=0.70, zorder=0)
        ax.axvline(n - 0.5, color=PALETTE["muted"], ls=(0, (4, 3)), lw=1.8, zorder=4)
        ax.set_xlim(0, horizon - 1)

    def _panel(self, ax: Axes, time_axis: bool = False) -> None:
        """
        Apply uniform gridlines, tick sizes, and optional integer time-index ticks.

        :param Axes ax: Target Matplotlib Axes panel.
        :param bool time_axis: If True, restricts x-axis major ticks to at most 4 integers.
        :return None: Modifies ax in place.
        """
        ax.set_axisbelow(True)
        ax.grid(True, which="major")
        ax.tick_params(axis="both", labelsize=16.5, length=5)
        if time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    def _pca_axis(
        self,
        ax: Axes,
        cum: np.ndarray,
        d95: int,
        d99: int,
        max_components: int,
    ) -> None:
        """
        Plot the per-window cumulative explained PCA variance curve with 95% and 99% thresholds.

        :param Axes ax: Target Matplotlib Axes panel.
        :param np.ndarray cum: Cumulative explained-variance ratios of shape (min(T, d_e),).
        :param int d95: Minimum number of principal components reaching 95% explained variance.
        :param int d99: Minimum number of principal components reaching 99% explained variance.
        :param int max_components: Maximum number of leading components to display on the x-axis.
        :return None: Draws the curve and threshold lines onto ax in place.
        """
        show = min(max_components, cum.size)
        k = np.arange(1, show + 1)
        y = cum[:show]

        ax.plot(k, y, color=PALETTE["blue"], lw=4.0, marker="o", markersize=8.0,
                markerfacecolor="white", markeredgewidth=2.2, zorder=4)
        ax.axhline(.95, color=PALETTE["orange"], ls="--", lw=3.2, zorder=3)
        ax.axhline(.99, color=PALETTE["magenta"], ls=(0, (3, 2)), lw=3.2, zorder=3)

        ylo, yhi = self._finite_limits(np.r_[y, .95, .99], floor=0., ceiling=1.005,
                                       relative_pad=.06, min_span=.03)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(.65, show + .4)
        ax.set_xticks(k)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100*value:.0f}%"))
        ax.set_title(f"PCA · 95%: {d95} és 99%: {d99} komp.", pad=12)
        ax.set_xlabel("Főkomponensek száma")
        ax.set_ylabel("Kumulatív variancia")
        self._panel(ax)

    def _cosine_axis(
        self,
        ax: Axes,
        values: np.ndarray,
        times: np.ndarray,
        n: int,
        horizon: int,
    ) -> None:
        """
        Plot pointwise Time-Feature cosine similarity over time with dynamic y-range annotation.

        :param Axes ax: Target Matplotlib Axes panel.
        :param np.ndarray values: Per-timestep cosine similarities of shape (T,) in [-1, 1].
        :param np.ndarray times: Discrete time index array of shape (T,).
        :param int n: Observation window length for shading the observed prefix.
        :param int horizon: Total trajectory length T.
        :return None: Draws the cosine-similarity curve onto ax in place.
        """
        ax.plot(times, values, color=PALETTE["navy"], lw=3.4,
                solid_joinstyle="round", zorder=3)
        self._mark_observed(ax, n, horizon)
        ylim = self._finite_limits(values, floor=-1., ceiling=1., relative_pad=.09, min_span=.0005)
        ax.set_ylim(*ylim)
        ax.set_title("Time–Feature · koszinuszos has.", pad=12)
        ax.set_xlabel("Időindex")
        ax.set_ylabel("Koszinuszos has.")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        self._panel(ax, time_axis=True)

        # Increase decimal precision when cosine similarity varies within a narrow band (< 0.03).
        digits = 5 if (ylim[1] - ylim[0]) < .03 else 3
        ax.text(.98, .04, f"{ylim[0]:.{digits}f} – {ylim[1]:.{digits}f}", ha="right", va="bottom",
                transform=ax.transAxes, fontsize=14.5, color=PALETTE["muted"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=3.0))

    def _procrustes_bars(self, ax: Axes, exp: Experiment) -> None:
        """
        Draw a horizontal bar chart of full-dimensional Procrustes disparity across all windows.

        :param Axes ax: Target Matplotlib Axes panel.
        :param Experiment exp: Loaded single-model Experiment containing observations (4,)
            and disparity (4,).
        :return None: Draws labeled horizontal bars onto ax in place.
        """
        observations = exp.observations.astype(int)
        values = exp.disparity
        ypos = np.arange(len(observations))
        colors = list(PROCRUSTES_COLORS[:len(observations)])

        valid = values[np.isfinite(values) & (values >= 0)]
        xmax = max(float(valid.max()) * 1.32 if valid.size else 0., .01)

        ax.barh(ypos, np.where(np.isfinite(values), values, 0.),
                height=.62, color=colors, alpha=.92)
        ax.set_yticks(ypos, labels=[f"{n} megfigyelés" for n in observations])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Procrustes-eltérés (kisebb érték: jobb geometriai illeszkedés)", fontsize=19)
        ax.set_title("A Time és Feature beágyazások Procrustes-eltérése",
                     pad=16, fontsize=22)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=16.5)
        ax.tick_params(axis="y", length=0, labelsize=18)

        # Annotate each bar with its exact 6-decimal Procrustes disparity value.
        for i, value in enumerate(values):
            label = f"{value:.6f}" if np.isfinite(value) else "nem számolható"
            x = float(value) if np.isfinite(value) else 0.
            ax.text(x + xmax * .014, i, label, va="center", ha="left",
                    fontsize=18.5, fontweight="bold", color=PALETTE["ink"])

    def _save(self, fig: Figure, destination: Path) -> None:
        """
        Save a figure as both raster PNG (at PNG_DPI) and vector PDF, then close it.

        :param Figure fig: Matplotlib Figure to persist.
        :param Path destination: Target PNG file path; a sibling .pdf file is written alongside it.
        :return None: Writes files to disk and releases figure memory.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=PNG_DPI, bbox_inches="tight", pad_inches=.22)
        fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.22)
        plt.close(fig)

    def plot_grid(self, exp: Experiment, output_dir: Path, pca_components: int) -> Path:
        """
        Render the 4x4+1 single-model diagnostic report (thesis Figures 11-14, 20-21).

        Each of the top 4 rows corresponds to one observation window and plots:
        (col 0) reference vs. predicted target with out-of-window MSE,
        (col 1) per-window cumulative PCA explained variance,
        (col 2) 2D Feature latent trajectory in the shared PCA basis, and
        (col 3) pointwise Time-Feature cosine similarity.
        The bottom row spans all 4 columns with Procrustes disparity bars.

        :param Experiment exp: Loaded Experiment with 4 observation windows and derived diagnostics.
        :param Path output_dir: Directory where the PNG and PDF reports are saved.
        :param int pca_components: Maximum number of principal components shown in column 1.
        :return Path: Path to the saved PNG report.
        """
        self.style()
        horizon = len(exp.target)
        times = np.arange(horizon)

        fig = plt.figure(figsize=(28.0, 22.0), facecolor="white")
        grid = fig.add_gridspec(5, 4,
                                height_ratios=[1, 1, 1, 1, .92],
                                left=.078, right=.982, top=.845, bottom=.055,
                                wspace=.40, hspace=.78)
        fig.suptitle(
            f"EINN · járványdinamikai és látens-térbeli elemzés   |   {exp.name}   |   $d_e={exp.dimension}$",
            fontsize=32, fontweight="bold", color=PALETTE["ink"], y=.982,
        )
        fig.text(.5, .938, "Négy megfigyelési ablak · azonos teszttrajektória · közös PCA-bázis",
                 ha="center", fontsize=21, color=PALETTE["muted"])

        # Top legends: forecast line styles (left) and PCA trajectory markers (right).
        pred_handles = [
            Line2D([0], [0], color="#171717", lw=3.5, label="Zajmentes referencia"),
            Line2D([0], [0], color=PALETTE["teal"], lw=3.5, label="Modellbecslés"),
            Patch(color=PALETTE["background_obs"], label="Megfigyelt időszak"),
        ]
        pca_handles = [
            Line2D([0], [0], lw=0, marker="x", color=PALETTE["ink"], ms=12, mew=2.6, label="Kezdet"),
            Line2D([0], [0], lw=0, marker="s", color=PALETTE["orange"], ms=12, label="Végpont"),
            Line2D([0], [0], lw=0, marker="o", markerfacecolor="none",
                   markeredgecolor=PALETTE["magenta"], ms=13, mew=2.6, label="Megfigyelés vége"),
        ]
        fig.legend(handles=pred_handles, loc="upper left", bbox_to_anchor=(.055, .915),
                   ncol=3, frameon=False, fontsize=18.5)
        fig.legend(handles=pca_handles, loc="upper right", bbox_to_anchor=(.980, .915),
                   ncol=3, frameon=False, fontsize=18.5)

        # Share vertical limits across all 4 forecast panels for direct cross-window comparison.
        ypred = self._finite_limits(np.concatenate([exp.target, exp.predictions.ravel()]),
                                    relative_pad=.08, min_span=.01)

        for i, obs in enumerate(exp.observations):
            n = int(obs)

            # Column 0: Noise-free reference vs. Feature-branch forecast and out-of-window MSE.
            ax = fig.add_subplot(grid[i, 0])
            ax.plot(times, exp.target, color="#171717", lw=3.0, zorder=3)
            ax.plot(times, exp.predictions[i], color=PALETTE["teal"], lw=3.2, zorder=3)
            self._mark_observed(ax, n, horizon)
            ax.set_ylim(*ypred)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(f"{n} megfigyelés · MSE: {exp.mse_future[i]:.3g}", pad=12)
            ax.set_xlabel("Időindex")
            ax.set_ylabel("Kumulált fertőzöttek" if exp.name == "ROST" else "Fertőzöttek aránya")
            self._panel(ax, time_axis=True)

            # Column 1: Per-window cumulative explained PCA variance.
            ax = fig.add_subplot(grid[i, 1])
            self._pca_axis(ax, exp.cumulative_variance[i], int(exp.d95[i]), int(exp.d99[i]), pca_components)

            # Column 2: 2D Feature latent trajectory (T, 2) in the shared PCA basis.
            ax = fig.add_subplot(grid[i, 2])
            coords = exp.pca_coords[i]
            ax.plot(coords[:, 0], coords[:, 1], color=PALETTE["gray"], lw=1.8, alpha=.58, zorder=1)
            ax.scatter(coords[:, 0], coords[:, 1], s=24, color=PALETTE["blue"],
                       alpha=.94, edgecolors="none", zorder=2)
            ax.scatter(*coords[0], marker="x", color=PALETTE["ink"], s=95, lw=2.5, zorder=4)
            ax.scatter(*coords[-1], marker="s", color=PALETTE["orange"], s=95, zorder=5)
            ax.scatter(*coords[n - 1], marker="o", facecolors="none",
                       edgecolors=PALETTE["magenta"], lw=2.8, s=175, zorder=6)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title("Feature-látens pálya · PCA", pad=12)
            ax.set_xlabel(f"1. főkomp. ({exp.shared_variance_ratio[0]:.1%})")
            ax.set_ylabel(f"2. főkomp. ({exp.shared_variance_ratio[1]:.1%})")
            self._panel(ax)

            # Column 3: Pointwise Time-Feature cosine similarity over time.
            ax = fig.add_subplot(grid[i, 3])
            self._cosine_axis(ax, exp.cosine[i], times, n, horizon)

        # Bottom row: Full-dimensional Procrustes disparity across all 4 windows.
        bottom = fig.add_subplot(grid[4, :])
        self._procrustes_bars(bottom, exp)

        destination = output_dir / f"{exp.name.lower()}_altalanositas_de_{exp.dimension}_4x4.png"
        self._save(fig, destination)
        return destination

    def plot_cosine_detail(
        self,
        exp: Experiment,
        directory: Path,
        cosine_range: tuple[int, int] | None,
    ) -> Path:
        """
        Render a full-width 4-row cosine-similarity figure with independent per-row y-scales.

        :param Experiment exp: Loaded Experiment containing cosine similarities of shape (4, T).
        :param Path directory: Output directory where PNG and PDF files are written.
        :param tuple[int, int] | None cosine_range: Optional half-open time slice [lo, hi) to zoom into;
            plots the full horizon [0, T) when None.
        :return Path: Path to the saved PNG figure.
        :raises ValueError: If cosine_range does not satisfy 0 <= lo < hi <= T.
        """
        self.style()
        horizon = exp.target.size
        if cosine_range is None:
            lo, hi = 0, horizon
        else:
            lo, hi = cosine_range
            if not (0 <= lo < hi <= horizon):
                raise ValueError(f"Hibás --cosine-range {cosine_range}, megengedett: [0, {horizon}].")

        ticks = np.arange(lo, hi)
        fig, axs = plt.subplots(4, 1, figsize=(17.0, 14.5), sharex=True)
        fig.subplots_adjust(left=.135, right=.975, top=.84, bottom=.08, hspace=.54)
        fig.suptitle(f"Koszinuszos hasonlóság részletesen   |   {exp.name}, $d_e={exp.dimension}$",
                     y=.98, fontsize=28, fontweight="bold", color=PALETTE["ink"])
        fig.text(.5, .932,
                 f"Vizsgált időpontok: {lo}–{hi-1}; az egyes sorok függőleges skálái különbözhetnek",
                 ha="center", fontsize=19, color=PALETTE["muted"])

        for i, (ax, obs) in enumerate(zip(axs, exp.observations)):
            n = int(obs)
            values = exp.cosine[i, lo:hi]
            ax.plot(ticks, values, lw=3.3, color=PALETTE["navy"])
            # Shade the observed portion only if the cutoff falls inside the requested zoom interval.
            if lo < n - .5 < hi:
                ax.axvspan(lo, n - .5, color=PALETTE["background_obs"], alpha=.8)
                ax.axvline(n - .5, lw=1.8, ls="--", color=PALETTE["muted"])
            ax.set_xlim(lo, hi - 1)
            ax.set_ylim(*self._finite_limits(values, floor=-1., ceiling=1., relative_pad=.12, min_span=.0003))
            ax.set_ylabel("Koszinuszos\nhasonlóság")
            ax.set_title(f"{n} megfigyelt időpont", loc="left", pad=10, fontsize=19.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            self._panel(ax)

        axs[-1].set_xlabel("Időindex")
        axs[-1].xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        destination = directory / f"{exp.name.lower()}_koszinusz_nagyitva_de_{exp.dimension}.png"
        self._save(fig, destination)
        return destination

    def plot_procrustes_detail(self, exp: Experiment, directory: Path) -> Path:
        """
        Render a standalone horizontal bar chart of Time-Feature Procrustes disparities.

        :param Experiment exp: Loaded Experiment containing disparity values of shape (4,).
        :param Path directory: Output directory where PNG and PDF files are written.
        :return Path: Path to the saved PNG figure.
        """
        self.style()
        fig, ax = plt.subplots(figsize=(14.5, 6.8))
        fig.subplots_adjust(left=.26, right=.95, top=.76, bottom=.20)
        fig.suptitle(f"Procrustes-összehasonlítás   |   {exp.name}, $d_e={exp.dimension}$",
                     fontsize=26, fontweight="bold", y=.96, color=PALETTE["ink"])
        self._procrustes_bars(ax, exp)
        destination = directory / f"{exp.name.lower()}_procrustes_de_{exp.dimension}.png"
        self._save(fig, destination)
        return destination

    def save_summary_csv(self, exp: Experiment, directory: Path) -> Path:
        """
        Export per-window forecasting and latent-space summary metrics to a CSV table.

        Writes one row per observation window with columns: observed_steps,
        future_mse_physical, pca_d95, pca_d99, procrustes_disparity,
        mean_cosine_similarity, pca_pc1_shared, and pca_pc2_shared.

        :param Experiment exp: Loaded Experiment containing all computed diagnostics.
        :param Path directory: Output directory where uj_abrak_numerikus_osszefoglalo.csv is written.
        :return Path: Path to the written CSV file.
        """
        path = directory / "uj_abrak_numerikus_osszefoglalo.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(("observed_steps", "future_mse_physical", "pca_d95", "pca_d99",
                             "procrustes_disparity", "mean_cosine_similarity", "pca_pc1_shared", "pca_pc2_shared"))
            for i, n in enumerate(exp.observations):
                sim = exp.cosine[i]
                valid = sim[np.isfinite(sim)]
                writer.writerow((int(n), float(exp.mse_future[i]), int(exp.d95[i]),
                                 int(exp.d99[i]), float(exp.disparity[i]),
                                 float(valid.mean()) if valid.size else float("nan"),
                                 float(exp.shared_variance_ratio[0]), float(exp.shared_variance_ratio[1])))
        return path
