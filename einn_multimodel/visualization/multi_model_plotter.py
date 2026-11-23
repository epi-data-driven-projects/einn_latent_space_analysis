"""Original multi-model 4x4 and joint-space report renderer."""
from __future__ import annotations
import os
from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.spatial import procrustes, cKDTree

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


class MultiModelPlotter:
    """Draw publication reports from a FeatureOnlyEvaluator without changing its predictions."""

    def __init__(self, evaluator):
        """Reference the existing evaluator and current experiment settings.

        :param Any evaluator: Evaluator whose arrays are to be visualized.
        :return None: None; the plotter stores its data source.
        """
        self.evaluator = evaluator
        self.experiment = evaluator.experiment

    def __getattr__(self, name):
        """Delegate read-only access to the evaluator's metrics and trained models.

        :param str name: Name of a source attribute.
        :return Any: Attribute value from the data source.
        """
        return getattr(self.evaluator, name)

    def configure_plot_style(self) -> None:
        """Exact visual language of the single-model 4x4 reference script.

        :return None: No return value; state or serialized artifacts are updated in place.
        """
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
    def _finite_limits(self, values, *, floor=None, ceiling=None,
                       relative_pad=.12, min_span=.015):
        """Determine common finite axis bounds with stable padding across compared trajectories and observation windows.


        :param Any values: Numerical values to summarize, normalize, or visualize.
        :param Any floor: Optional lower limit of the displayed numerical axis.
        :param Any ceiling: Optional upper limit of the displayed numerical axis.
        :param Any relative_pad: Relative padding added around the observed numerical range.
        :param Any min_span: Lower bound on plot range to avoid degenerate axes.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
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
    def _panel_style(self, ax, time_axis=False):
        """Apply a consistent readable visual style to one Matplotlib axes.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param Any time_axis: Whether the panel should use the common simulation time-axis formatting.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        ax.set_axisbelow(True)
        ax.grid(True, which="major")
        ax.tick_params(axis="both", labelsize=9, length=3)
        if time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    def _mark_observed(self, ax, n, horizon):
        """Shade the observed prefix and draw its cutoff in a forecast panel.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param Any n: Number of observations, components, or plot elements requested.
        :param Any horizon: Total number of simulation time points in the trajectory.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        ax.axvspan(-.5, n - .5, color=PALETTE["background_obs"], alpha=.70, zorder=0)
        ax.axvline(n - .5, color=PALETTE["muted"], ls=(0, (4, 3)), lw=1.4, zorder=4)
        ax.set_xlim(0, horizon - 1)
    def _pca_axis(self, ax, cum, d95, d99, max_components):
        """Format a principal-component projection using shared plot limits.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param Any cum: Cumulative explained-variance proportions of the PCA components.
        :param Any d95: Minimum PCA component count exceeding 95 percent explained variance.
        :param Any d99: Minimum PCA component count exceeding 99 percent explained variance.
        :param Any max_components: Maximum number of principal components to display.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        show = min(max_components, cum.size)
        x = np.arange(1, show + 1)
        y = cum[:show]
        ax.plot(x, y, color=PALETTE["blue"], lw=3.6, marker="o", markersize=6.4,
                markerfacecolor="white", markeredgewidth=1.8, zorder=4)
        ax.axhline(.95, color=PALETTE["orange"], ls="--", lw=2.8, zorder=3)
        ax.axhline(.99, color=PALETTE["magenta"], ls=(0, (3, 2)), lw=2.8, zorder=3)
        ax.set_ylim(*self._finite_limits(np.r_[y, .95, .99], floor=0., ceiling=1.005,
                                    relative_pad=.06, min_span=.03))
        ax.set_xlim(.65, show + .4)
        ax.set_xticks(x)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
        ax.set_title(f"PCA · 95%: {d95} és 99%: {d99} komponens", pad=9)
        ax.set_xlabel("Főkomponensek száma")
        ax.set_ylabel("Kumulatív variancia")
        self._panel_style(ax)
    def _cosine_axis(self, ax, values, times, n, horizon):
        """Configure the scale and labels of cosine-similarity plots.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param Any values: Numerical values to summarize, normalize, or visualize.
        :param Any times: Time coordinates corresponding to the sampled states or embeddings.
        :param Any n: Number of observations, components, or plot elements requested.
        :param Any horizon: Total number of simulation time points in the trajectory.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        ax.plot(times, values, color=PALETTE["navy"], lw=3.1,
                solid_joinstyle="round", zorder=3)
        self._mark_observed(ax, n, horizon)
        limits = self._finite_limits(values, floor=-1., ceiling=1.,
                                relative_pad=.09, min_span=.0005)
        ax.set_ylim(*limits)
        ax.set_title("Time–Feature · koszinuszos hasonlóság", pad=9)
        ax.set_xlabel("Időindex")
        ax.set_ylabel("Koszinuszos hasonlóság")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        self._panel_style(ax, time_axis=True)
        digits = 5 if (limits[1] - limits[0]) < .03 else 3
        ax.text(.98, .03, f"{limits[0]:.{digits}f} – {limits[1]:.{digits}f}",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=8,
                color=PALETTE["muted"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=.75, pad=2))
    def _procrustes_bars(self, ax, phase_results):
        """Display the geometric discrepancy for the compared observation windows.


        :param Any ax: Matplotlib axes on which the diagnostic plot is drawn.
        :param Any phase_results: Predictions and embedding arrays for every selected observation window.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
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
    def _save_plot(self, fig, path):
        """Persist a report in PNG and PDF formats without changing its data.


        :param Any fig: Matplotlib figure containing the report panels.
        :param Any path: File-system path to the experiment artifact or configuration.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=.18)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.18)
        plt.close(fig)
    def plot_multimodel_generalization_4x4(self, phase_results, shared_pca,
                                           output_path, hausdorff_metrics):
        """Same layout, colors, legend positions and scaling as single-model plot.

        :param Any phase_results: Predictions and embedding arrays for every selected observation window.
        :param Any shared_pca: PCA estimator already fitted to the combined latent embeddings.
        :param Any output_path: Destination path for the rendered figure.
        :param Any hausdorff_metrics: Original-dimensional Hausdorff distances and related summary metrics.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        if len(phase_results) != 4:
            raise ValueError("Exactly four observation windows are required.")
        self.configure_plot_style()
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
            f"{model_name}   |   $d_e={self.experiment.EMBEDDING_DIM}$",
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
        ypred = self._finite_limits(np.r_[actual, predicted],
                               relative_pad=.08, min_span=.01)

        for row, r in enumerate(phase_results):
            n = int(r["t_end"])
            ax = fig.add_subplot(grid[row, 0])
            ax.plot(times, actual, color="#171717", lw=2.6, zorder=3)
            ax.plot(times, r["predicted_trajectory"],
                    color=PALETTE["teal"], lw=2.75, zorder=3)
            self._mark_observed(ax, n, horizon)
            ax.set_ylim(*ypred)
            ax.set_title(f"{n} megfigyelés · jövőbeli MSE: {r['mse_future']:.3g}", pad=9)
            ax.set_xlabel("Időindex")
            ax.set_ylabel("Fertőzöttek aránya")
            self._panel_style(ax, time_axis=True)

            ax = fig.add_subplot(grid[row, 1])
            self._pca_axis(ax, r["cum_variance"], r["pca_95"], r["pca_99"],
                      self.experiment.PCA_COMPONENTS_TO_SHOW)

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
            self._panel_style(ax)

            ax = fig.add_subplot(grid[row, 3])
            self._cosine_axis(ax, r["cosine_similarity"], times, n, horizon)

        ax_procrustes = fig.add_subplot(grid[4, :])
        self._procrustes_bars(ax_procrustes, phase_results)
        self._save_plot(fig, output_path)
    def plot_joint_latent_space_hausdorff(self, data, metrics):
        """All independent held-out trajectories in ONE common PCA coordinate system.

        Hausdorff bars are computed in ORIGINAL d_e, never on the 2D projection.
        The central map shows temporal Feature curves and uses model colors.

        :param Any data: Data mapping or array to be processed by the operation.
        :param Any metrics: Training, forecast, or latent-space diagnostic measurements.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        self.configure_plot_style()
        E = np.asarray(data["feature_embeddings"], dtype=float)
        labels = np.asarray(data["true_labels"], dtype=int)
        trajectory_ids = np.asarray(data["trajectory_ids"], dtype=int)
        if E.ndim != 2 or E.shape[0] < 3:
            raise ValueError(f"Invalid held-out embedding matrix: {E.shape}")
        if not np.isfinite(E).all():
            raise ValueError("Joint latent PCA: non-finite embeddings")

        shared = PCA(n_components=2)
        coords = shared.fit_transform(E)
        model_a, model_b = self.experiment.MODEL_NAMES
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
            f"{model_a} / {model_b} | $d_e={self.experiment.EMBEDDING_DIM}$",
            fontsize=20, color=PALETTE["ink"], fontweight="bold", y=.982,
        )
        fig.text(.5, .938,
                 "Az összes független teszttrajektória · egyetlen közös PCA-bázis · "
                 "pontok időrendben összekötve",
                 fontsize=11.5, color=PALETTE["muted"], ha="center")

        # The joint PCA map is deliberately the dominant element in the figure.
        ax = fig.add_subplot(grid[:, :3])
        class_counts = {}
        for model_name in self.experiment.MODEL_NAMES:
            class_id = self.models.model_to_id[model_name]
            class_counts[model_name] = int(np.count_nonzero([
                int(self.dataset.true_model_labels[i]) == class_id
                for i in range(len(self.dataset.true_model_labels))
            ]))
        for model_name in self.experiment.MODEL_NAMES:
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
            for model in self.experiment.MODEL_NAMES
        ], loc="upper right", frameon=True, framealpha=.95, fontsize=11)
        self._panel_style(ax)
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
            f"{model_b.lower()}_de_{self.experiment.EMBEDDING_DIM}.png",
        )
        self._save_plot(fig, destination)

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
