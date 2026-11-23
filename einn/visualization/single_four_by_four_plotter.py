"""Saved-result presentation settings preserved from the original script."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from einn.model.interface.saved_experiment import Experiment
import warnings
import csv


DEFAULT_ROOT = Path("./experiment_results/generalization")

RUN_ALL_MODELS = True

ALL_MODEL_NAMES = ("SIR", "SIS", "ROST")

ALL_EMBEDDING_DIMS = (20, 5)

OUTPUT_SUBDIRECTORY = "ujrarajzolt_abrak"

DEFAULT_PCA_COMPONENTS = 8

DEFAULT_OBSERVATION_WINDOWS = (10, 18, 40, 80)

PNG_DPI = 230

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

PROCRUSTES_COLORS = (
    "#3B7A57",  # Forest green, visually distinct from the main report curves.
    "#6B4E9B",  # Deep purple, visually distinct from the main report curves.
    "#C44536",  # Muted vermilion, visually distinct from the main report curves.
    "#8C6D46",  # warm bronze
)




class SingleModelFourByFourPlotter:
    """Render the original experiment reports without changing numerical results."""

    def style(self) -> None:
        """Apply the established report font family, sizes, colors, and grid settings.


        :return None: No return value; state or serialized artifacts are updated in place.
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
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })

    def _finite_limits(self, values: np.ndarray, *, floor: float | None = None,
                       ceiling: float | None = None, relative_pad: float = 0.12,
                       min_span: float = 0.015) -> tuple[float, float]:
        """Determine common finite axis bounds with stable padding across compared trajectories and observation windows.


        :param np.ndarray values: Numerical values to summarize, normalize, or visualize.
        :param float | None floor: Optional lower limit of the displayed numerical axis.
        :param float | None ceiling: Optional upper limit of the displayed numerical axis.
        :param float relative_pad: Relative padding added around the observed numerical range.
        :param float min_span: Lower bound on plot range to avoid degenerate axes.
        :return tuple[float, float]: Computed result in the configured output representation.
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
        if high <= low:
            middle = float(np.mean(vals))
            low = max(floor, middle - .02) if floor is not None else middle - .02
            high = min(ceiling, middle + .02) if ceiling is not None else middle + .02
        return low, high

    def _mark_observed(self, ax, n: int, horizon: int):
        """Shade the observed prefix and draw its cutoff in a forecast panel.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param int n: Number of observations, components, or plot elements requested.
        :param int horizon: Total number of simulation time points in the trajectory.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        ax.axvspan(-0.5, n - 0.5, color=PALETTE["background_obs"], alpha=0.70, zorder=0)
        ax.axvline(n - 0.5, color=PALETTE["muted"], ls=(0, (4, 3)), lw=1.8, zorder=4)
        ax.set_xlim(0, horizon - 1)

    def _panel(self, ax, time_axis: bool = False):
        """Apply common panel styling and optional temporal-axis labels.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param bool time_axis: Whether the panel should use the common simulation time-axis formatting.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        ax.set_axisbelow(True)
        ax.grid(True, which="major")
        ax.tick_params(axis="both", labelsize=16.5, length=5)
        if time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    def _pca_axis(self, ax, cum: np.ndarray, d95: int, d99: int, max_components: int):
        """Format a principal-component projection using shared plot limits.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param np.ndarray cum: Cumulative explained-variance proportions of the PCA components.
        :param int d95: Minimum PCA component count exceeding 95 percent explained variance.
        :param int d99: Minimum PCA component count exceeding 99 percent explained variance.
        :param int max_components: Maximum number of principal components to display.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
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

    def _cosine_axis(self, ax, values: np.ndarray, times: np.ndarray, n: int, horizon: int):
        """Configure the scale and labels of cosine-similarity plots.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param np.ndarray values: Numerical values to summarize, normalize, or visualize.
        :param np.ndarray times: Time coordinates corresponding to the sampled states or embeddings.
        :param int n: Number of observations, components, or plot elements requested.
        :param int horizon: Total number of simulation time points in the trajectory.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
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
        digits = 5 if (ylim[1] - ylim[0]) < .03 else 3
        ax.text(.98, .04, f"{ylim[0]:.{digits}f} – {ylim[1]:.{digits}f}", ha="right", va="bottom",
                transform=ax.transAxes, fontsize=14.5, color=PALETTE["muted"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=3.0))

    def _procrustes_bars(self, ax, exp: Experiment):
        """Display the geometric discrepancy for the compared observation windows.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param Experiment exp: Experiment settings and previously computed observations used for rendering.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
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
        for i, value in enumerate(values):
            label = f"{value:.6f}" if np.isfinite(value) else "nem számolható"
            x = float(value) if np.isfinite(value) else 0.
            ax.text(x + xmax * .014, i, label, va="center", ha="left",
                    fontsize=18.5, fontweight="bold", color=PALETTE["ink"])

    def _save(self, fig, destination: Path):
        """Persist a plot, fitted model component, or experiment output to its configured path.


        :param Any fig: Matplotlib figure containing the report panels.
        :param Path destination: Path of the newly generated report or persisted data artifact.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=PNG_DPI, bbox_inches="tight", pad_inches=.22)
        fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.22)
        plt.close(fig)

    def plot_grid(self, exp: Experiment, output_dir: Path, pca_components: int) -> Path:
        """Render the original grid of numerical and latent-space diagnostics.


        :param Experiment exp: Experiment settings and previously computed observations used for rendering.
        :param Path output_dir: Directory where generated results are written.
        :param int pca_components: Number of principal components retained in a visualization.
        :return Path: Computed result in the configured output representation.
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

        ypred = self._finite_limits(np.concatenate([exp.target, exp.predictions.ravel()]),
                               relative_pad=.08, min_span=.01)
        for i, obs in enumerate(exp.observations):
            n = int(obs)
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

            ax = fig.add_subplot(grid[i, 1])
            self._pca_axis(ax, exp.cumulative_variance[i], int(exp.d95[i]), int(exp.d99[i]), pca_components)

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

            ax = fig.add_subplot(grid[i, 3])
            self._cosine_axis(ax, exp.cosine[i], times, n, horizon)

        bottom = fig.add_subplot(grid[4, :])
        self._procrustes_bars(bottom, exp)
        destination = output_dir / f"{exp.name.lower()}_altalanositas_de_{exp.dimension}_4x4.png"
        self._save(fig, destination)
        return destination

    def plot_cosine_detail(self, exp: Experiment, directory: Path,
                           cosine_range: tuple[int, int] | None) -> Path:
        """Draw the full-width cosine-similarity diagnostics using separately scaled observation-window panels.


        :param Experiment exp: Experiment settings and previously computed observations used for rendering.
        :param Path directory: Directory containing the saved input data or receiving report files.
        :param tuple[int, int] | None cosine_range: Optional common scale for cosine-similarity diagnostics.
        :return Path: Computed result in the configured output representation.
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
        """Render the expanded Procrustes comparison report.


        :param Experiment exp: Experiment settings and previously computed observations used for rendering.
        :param Path directory: Directory containing the saved input data or receiving report files.
        :return Path: Computed result in the configured output representation.
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
        """Persist save summary csv using the original serialization format.


        :param Experiment exp: Experiment settings and previously computed observations used for rendering.
        :param Path directory: Directory containing the saved input data or receiving report files.
        :return Path: Computed result in the configured output representation.
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
