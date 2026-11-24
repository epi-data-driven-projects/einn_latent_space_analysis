"""
Publication report renderer for multi-model EINN forecasting and latent-space diagnostics.

Generates the 4x4 per-window generalization/PCA/cosine report plus Procrustes disparity
bar chart, and the joint held-out Feature latent-space 2D PCA map accompanied by
original-space Hausdorff distance metrics and compressed `.npz` exports (thesis Sections 4-5).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from sklearn.decomposition import PCA

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

PNG_DPI: int = 230


class MultiModelPlotter:
    """
    Renderer for multi-model EINN evaluation figures and latent-space artifact archives.

    Delegates attribute lookups for evaluator arrays, metrics, and trained models to the
    wrapped evaluator without altering underlying predictions or computed distances.
    """

    def __init__(self, evaluator: Any) -> None:
        """
        Bind the plotter to an evaluator instance and its experiment configuration.

        :param Any evaluator: Evaluator exposing predictions, embeddings, metrics, output_dir,
            dataset, models, and experiment settings.
        :return None: Stores evaluator and experiment references in place.
        """
        self.evaluator: Any = evaluator
        self.experiment: Any = evaluator.experiment

    def __getattr__(self, name: str) -> Any:
        """
        Delegate missing attribute reads to the underlying evaluator.

        :param str name: Attribute name not defined directly on MultiModelPlotter.
        :return Any: Attribute value retrieved from self.evaluator.
        """
        return getattr(self.evaluator, name)

    def configure_plot_style(self) -> None:
        """
        Apply consistent Matplotlib typography, spine, grid, and vector-font defaults.

        :return None: Mutates plt.rcParams for publication-grade PNG and PDF output.
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

    def _finite_limits(
        self,
        values: Any,
        *,
        floor: float | None = None,
        ceiling: float | None = None,
        relative_pad: float = 0.12,
        min_span: float = 0.015,
    ) -> tuple[float, float]:
        """
        Compute padded finite axis limits, clamping to optional floor/ceiling bounds.

        :param Any values: Array-like numerical values used to determine axis range.
        :param float | None floor: Optional minimum allowed lower axis bound.
        :param float | None ceiling: Optional maximum allowed upper axis bound.
        :param float relative_pad: Fractional margin added to the data span.
        :param float min_span: Minimum span enforced before padding to prevent flat axes.
        :return tuple[float, float]: Pair (lower_bound, upper_bound) for axis scaling.
        """
        v = np.asarray(values, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (-0.05, 1.05)

        lo, hi = float(v.min()), float(v.max())
        span = max(hi - lo, min_span)
        pad = max(span * relative_pad, min_span * 0.30)
        lo, hi = lo - pad, hi + pad
        if floor is not None:
            lo = max(lo, floor)
        if ceiling is not None:
            hi = min(hi, ceiling)

        # Ensure a non-zero axis interval if clamping collapsed lo and hi.
        if hi <= lo:
            mid = float(v.mean())
            lo = max(floor, mid - 0.02) if floor is not None else mid - 0.02
            hi = min(ceiling, mid + 0.02) if ceiling is not None else mid + 0.02
        return lo, hi

    def _panel_style(self, ax: plt.Axes, time_axis: bool = False) -> None:
        """
        Configure grid placement, tick sizes, and optional integer time-step ticks on an axes.

        :param plt.Axes ax: Target Matplotlib axes to style.
        :param bool time_axis: If True, formats the x-axis with integer major ticks.
        :return None: Updates ax styling in place.
        """
        ax.set_axisbelow(True)
        ax.grid(True, which="major")
        ax.tick_params(axis="both", labelsize=9, length=3)
        if time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    def _mark_observed(self, ax: plt.Axes, n: int, horizon: int) -> None:
        """
        Highlight the observed prefix [0, n) and mark the forecast cutoff boundary on a time series axis.

        :param plt.Axes ax: Target Matplotlib time-series axes.
        :param int n: Number of observed timesteps in the current window prefix.
        :param int horizon: Total trajectory length along the time axis.
        :return None: Draws shaded observation span, cutoff line, and x-limits on ax.
        """
        ax.axvspan(-0.5, n - 0.5, color=PALETTE["background_obs"], alpha=0.70, zorder=0)
        ax.axvline(n - 0.5, color=PALETTE["muted"], ls=(0, (4, 3)), lw=1.4, zorder=4)
        ax.set_xlim(0, horizon - 1)

    def _pca_axis(
        self,
        ax: plt.Axes,
        cum: np.ndarray,
        d95: int,
        d99: int,
        max_components: int,
    ) -> None:
        """
        Plot cumulative explained variance with 95% and 99% threshold reference lines.

        :param plt.Axes ax: Target Matplotlib axes for the cumulative PCA variance curve.
        :param np.ndarray cum: 1D cumulative explained-variance ratio array of length d_e.
        :param int d95: Number of principal components needed to reach 95% cumulative variance.
        :param int d99: Number of principal components needed to reach 99% cumulative variance.
        :param int max_components: Maximum number of principal components shown on the x-axis.
        :return None: Renders cumulative variance curve, thresholds, title, and labels on ax.
        """
        show = min(max_components, cum.size)
        x = np.arange(1, show + 1)
        y = cum[:show]
        ax.plot(
            x,
            y,
            color=PALETTE["blue"],
            lw=3.6,
            marker="o",
            markersize=6.4,
            markerfacecolor="white",
            markeredgewidth=1.8,
            zorder=4,
        )
        ax.axhline(0.95, color=PALETTE["orange"], ls="--", lw=2.8, zorder=3)
        ax.axhline(0.99, color=PALETTE["magenta"], ls=(0, (3, 2)), lw=2.8, zorder=3)
        ax.set_ylim(*self._finite_limits(
            np.r_[y, 0.95, 0.99], floor=0.0, ceiling=1.005, relative_pad=0.06, min_span=0.03
        ))
        ax.set_xlim(0.65, show + 0.4)
        ax.set_xticks(x)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.set_title(f"PCA · 95%: {d95} és 99%: {d99} komponens", pad=9)
        ax.set_xlabel("Főkomponensek száma")
        ax.set_ylabel("Kumulatív variancia")
        self._panel_style(ax)

    def _cosine_axis(
        self,
        ax: plt.Axes,
        values: np.ndarray,
        times: np.ndarray,
        n: int,
        horizon: int,
    ) -> None:
        """
        Plot per-timestep cosine similarity between Time and Feature latent embeddings.

        :param plt.Axes ax: Target Matplotlib axes for the cosine similarity curve.
        :param np.ndarray values: Cosine similarity values across timesteps of shape (horizon,).
        :param np.ndarray times: Integer time indices of shape (horizon,).
        :param int n: Number of observed timesteps in the current window prefix.
        :param int horizon: Total trajectory horizon length.
        :return None: Renders cosine similarity curve, observed window marker, and range badge.
        """
        ax.plot(times, values, color=PALETTE["navy"], lw=3.1, solid_joinstyle="round", zorder=3)
        self._mark_observed(ax, n, horizon)
        limits = self._finite_limits(
            values, floor=-1.0, ceiling=1.0, relative_pad=0.09, min_span=0.0005
        )
        ax.set_ylim(*limits)
        ax.set_title("Time–Feature · koszinuszos hasonlóság", pad=9)
        ax.set_xlabel("Időindex")
        ax.set_ylabel("Koszinuszos hasonlóság")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        self._panel_style(ax, time_axis=True)

        # Adapt decimal precision to narrow similarity spans so small variations remain legible.
        digits = 5 if (limits[1] - limits[0]) < 0.03 else 3
        ax.text(
            0.98,
            0.03,
            f"{limits[0]:.{digits}f} – {limits[1]:.{digits}f}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color=PALETTE["muted"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        )

    def _procrustes_bars(self, ax: plt.Axes, phase_results: list[dict[str, Any]]) -> None:
        """
        Draw a horizontal bar chart of Procrustes disparities across observation windows.

        :param plt.Axes ax: Target Matplotlib axes spanning the bottom summary row.
        :param list[dict[str, Any]] phase_results: Per-window evaluation dictionaries containing
            't_end' and 'procrustes_disparity'.
        :return None: Renders horizontal bars and numeric annotations on ax.
        """
        obs = [int(r["t_end"]) for r in phase_results]
        vals = np.asarray([r["procrustes_disparity"] for r in phase_results], dtype=float)
        yp = np.arange(len(obs))
        colors = [PALETTE["teal_dark"], PALETTE["blue"], PALETTE["orange"], PALETTE["magenta"]]
        good = vals[np.isfinite(vals) & (vals >= 0)]
        xmax = max(float(good.max()) * 1.22 if good.size else 0.0, 0.01)

        ax.barh(yp, np.where(np.isfinite(vals), vals, 0), height=0.61, color=colors[:len(obs)], alpha=0.92)
        ax.set_yticks(yp, labels=[f"{n} megfigyelés" for n in obs])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Procrustes-eltérés (kisebb érték: jobb geometriai illeszkedés)")
        ax.set_title("A megfelelő Time és Feature beágyazások Procrustes-eltérése", pad=12, fontsize=13)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        for i, val in enumerate(vals):
            txt = f"{val:.6f}" if np.isfinite(val) else "nem számolható"
            ax.text(
                (float(val) if np.isfinite(val) else 0.0) + xmax * 0.011,
                i,
                txt,
                va="center",
                ha="left",
                fontsize=10.5,
                fontweight="bold",
                color=PALETTE["ink"],
            )

    def _save_plot(self, fig: plt.Figure, path: str | os.PathLike[str]) -> None:
        """
        Save a Matplotlib figure as both high-DPI PNG and vector PDF, then close the figure.

        :param plt.Figure fig: Figure instance to export.
        :param str | os.PathLike[str] path: Destination path (PNG extension or base path).
        :return None: Writes PNG and PDF files and closes fig to release memory.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.18)
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)

    def plot_multimodel_generalization_4x4(
        self,
        phase_results: list[dict[str, Any]],
        shared_pca: PCA,
        output_path: str | os.PathLike[str],
        hausdorff_metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Render the 4-window trajectory/PCA/cosine grid and bottom Procrustes disparity chart.

        Each of the four rows corresponds to one expanding-window observation cutoff n = t_end,
        displaying: (0) target compartment forecast vs. ground truth, (1) cumulative PCA variance,
        (2) 2D latent trajectory in the shared PCA basis, and (3) Time-Feature cosine similarity.

        :param list[dict[str, Any]] phase_results: Four-element list of per-window result dicts.
        :param PCA shared_pca: Fitted PCA estimator providing shared 2D basis and variance ratios.
        :param str | os.PathLike[str] output_path: Destination PNG path (PDF is saved alongside).
        :param dict[str, Any] | None hausdorff_metrics: Optional Hausdorff summary dictionary passed
            for API compatibility with evaluator callers.
        :return None: Writes the multi-panel figure to disk.
        :raises ValueError: If phase_results does not contain exactly four observation windows.
        """
        del hausdorff_metrics  # Preserved in signature for caller compatibility.
        if len(phase_results) != 4:
            raise ValueError("Exactly four observation windows are required.")

        self.configure_plot_style()
        fig = plt.figure(figsize=(26.5, 19.0), facecolor="white")
        grid = fig.add_gridspec(
            5,
            4,
            height_ratios=[1, 1, 1, 1, 0.84],
            left=0.060,
            right=0.982,
            top=0.875,
            bottom=0.052,
            wspace=0.32,
            hspace=0.57,
        )
        model_name = phase_results[0]["true_model_name"]
        horizon = len(phase_results[0]["true_target"])
        times = np.arange(horizon)
        actual = phase_results[0]["true_target"]

        fig.suptitle(
            f"Többmodelles EINN · járványdinamikai és látens-térbeli elemzés   |   "
            f"{model_name}   |   $d_e={self.experiment.EMBEDDING_DIM}$",
            fontsize=20,
            fontweight="bold",
            color=PALETTE["ink"],
            y=0.987,
        )
        fig.text(
            0.5,
            0.952,
            "Négy megfigyelési ablak · azonos teszttrajektória · "
            "megfelelő Time Module · közös PCA-bázis",
            ha="center",
            fontsize=12,
            color=PALETTE["muted"],
        )

        pred_handles = [
            Line2D([0], [0], color="#171717", lw=2.6, label="Zajmentes referencia"),
            Line2D([0], [0], color=PALETTE["teal"], lw=2.6, label="Modellbecslés"),
            Patch(color=PALETTE["background_obs"], label="Megfigyelt időszak"),
        ]
        pca_handles = [
            Line2D([0], [0], lw=0, marker="x", color=PALETTE["ink"], ms=8, label="Kezdet"),
            Line2D([0], [0], lw=0, marker="s", color=PALETTE["orange"], ms=8, label="Végpont"),
            Line2D(
                [0],
                [0],
                lw=0,
                marker="o",
                markerfacecolor="none",
                markeredgecolor=PALETTE["magenta"],
                ms=9,
                label="Megfigyelés vége",
            ),
        ]
        fig.legend(
            handles=pred_handles,
            loc="upper left",
            bbox_to_anchor=(0.055, 0.935),
            ncol=3,
            frameon=False,
            fontsize=9.6,
        )
        fig.legend(
            handles=pca_handles,
            loc="upper right",
            bbox_to_anchor=(0.968, 0.935),
            ncol=3,
            frameon=False,
            fontsize=9.6,
        )

        predicted = np.concatenate([r["predicted_trajectory"] for r in phase_results])
        ypred = self._finite_limits(np.r_[actual, predicted], relative_pad=0.08, min_span=0.01)

        for row, r in enumerate(phase_results):
            n = int(r["t_end"])

            # Column 0: Forecast vs. noiseless reference with observed window shading.
            ax = fig.add_subplot(grid[row, 0])
            ax.plot(times, actual, color="#171717", lw=2.6, zorder=3)
            ax.plot(times, r["predicted_trajectory"], color=PALETTE["teal"], lw=2.75, zorder=3)
            self._mark_observed(ax, n, horizon)
            ax.set_ylim(*ypred)
            ax.set_title(f"{n} megfigyelés · jövőbeli MSE: {r['mse_future']:.3g}", pad=9)
            ax.set_xlabel("Időindex")
            ax.set_ylabel("Fertőzöttek aránya")
            self._panel_style(ax, time_axis=True)

            # Column 1: Cumulative PCA explained variance and 95%/99% component thresholds.
            ax = fig.add_subplot(grid[row, 1])
            self._pca_axis(
                ax,
                r["cum_variance"],
                r["pca_95"],
                r["pca_99"],
                self.experiment.PCA_COMPONENTS_TO_SHOW,
            )

            # Column 2: 2D trajectory in shared PCA space with start, end, and cutoff markers.
            ax = fig.add_subplot(grid[row, 2])
            coords = r["pca_coords"]
            ax.plot(coords[:, 0], coords[:, 1], color=PALETTE["gray"], lw=1.45, alpha=0.58, zorder=1)
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=16,
                color=PALETTE["blue"],
                alpha=0.94,
                edgecolors="none",
                zorder=2,
            )
            ax.scatter(*coords[0], marker="x", color=PALETTE["ink"], s=65, lw=1.8, zorder=4)
            ax.scatter(*coords[-1], marker="s", color=PALETTE["orange"], s=67, zorder=5)
            ax.scatter(
                *coords[n - 1],
                marker="o",
                facecolors="none",
                edgecolors=PALETTE["magenta"],
                lw=2.1,
                s=125,
                zorder=6,
            )
            ax.set_title("Feature-látens pálya · közös PCA", pad=9)
            ax.set_xlabel(f"1. főkomponens ({shared_pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"2. főkomponens ({shared_pca.explained_variance_ratio_[1]:.1%})")
            self._panel_style(ax)

            # Column 3: Cosine similarity between Time and Feature latent trajectories.
            ax = fig.add_subplot(grid[row, 3])
            self._cosine_axis(ax, r["cosine_similarity"], times, n, horizon)

        ax_procrustes = fig.add_subplot(grid[4, :])
        self._procrustes_bars(ax_procrustes, phase_results)
        self._save_plot(fig, output_path)

    def plot_joint_latent_space_hausdorff(
        self,
        data: dict[str, Any],
        metrics: dict[str, float],
    ) -> str:
        """
        Render the joint held-out Feature PCA map, raw-space Hausdorff metrics, and `.npz` export.

        Projects all held-out Feature embeddings into a single shared 2D PCA coordinate system
        for visualization while reporting Hausdorff distances computed in the full d_e-dimensional
        embedding space (thesis Sections 4 and 5).

        :param dict[str, Any] data: Dictionary containing 'feature_embeddings', 'true_labels',
            'trajectory_ids', and 'time_indices'.
        :param dict[str, float] metrics: Computed Hausdorff metrics including 'hausdorff',
            'directed_a_to_b', 'directed_b_to_a', 'hausdorff_p95',
            'hausdorff_unit_normalized', and 'hausdorff_p95_unit_normalized'.
        :return str: File path of the saved PNG figure.
        :raises ValueError: If feature_embeddings is not a 2D array with at least 3 rows or
            contains non-finite values.
        """
        self.configure_plot_style()
        embeddings = np.asarray(data["feature_embeddings"], dtype=float)
        labels = np.asarray(data["true_labels"], dtype=int)
        trajectory_ids = np.asarray(data["trajectory_ids"], dtype=int)
        if embeddings.ndim != 2 or embeddings.shape[0] < 3:
            raise ValueError(f"Invalid held-out embedding matrix: {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise ValueError("Joint latent PCA: non-finite embeddings")

        shared = PCA(n_components=2)
        coords = shared.fit_transform(embeddings)
        model_a, model_b = self.experiment.MODEL_NAMES
        model_colors = {
            model_a: PALETTE["teal"],
            model_b: PALETTE["magenta"],
        }

        fig = plt.figure(figsize=(19.0, 11.4), facecolor="white")
        grid = fig.add_gridspec(
            nrows=3,
            ncols=5,
            left=0.052,
            right=0.963,
            top=0.84,
            bottom=0.105,
            width_ratios=[1.0, 1.0, 1.0, 0.83, 0.83],
            height_ratios=[0.52, 2.0, 1.0],
            wspace=0.68,
            hspace=0.70,
        )
        fig.suptitle(
            f"EINN · közös Feature-látens tér és modellkülönbségek | "
            f"{model_a} / {model_b} | $d_e={self.experiment.EMBEDDING_DIM}$",
            fontsize=20,
            color=PALETTE["ink"],
            fontweight="bold",
            y=0.982,
        )
        fig.text(
            0.5,
            0.938,
            "Az összes független teszttrajektória · egyetlen közös PCA-bázis · "
            "pontok időrendben összekötve",
            fontsize=11.5,
            color=PALETTE["muted"],
            ha="center",
        )

        # Dominant left/center panel: all held-out trajectories in a single shared 2D PCA basis.
        ax = fig.add_subplot(grid[:, :3])
        class_counts = {
            model_name: int(
                np.count_nonzero([
                    int(self.dataset.true_model_labels[i]) == self.models.model_to_id[model_name]
                    for i in range(len(self.dataset.true_model_labels))
                ])
            )
            for model_name in self.experiment.MODEL_NAMES
        }
        for model_name in self.experiment.MODEL_NAMES:
            class_id = self.models.model_to_id[model_name]
            color = model_colors[model_name]
            for tr in np.unique(trajectory_ids[labels == class_id]):
                ix = np.flatnonzero((labels == class_id) & (trajectory_ids == tr))
                p = coords[ix]
                ax.plot(p[:, 0], p[:, 1], color=color, lw=1.75, alpha=0.47, zorder=2)
                ax.scatter(
                    p[:, 0],
                    p[:, 1],
                    color=color,
                    s=7.5,
                    alpha=0.28,
                    edgecolors="none",
                    zorder=3,
                    rasterized=True,
                )
                ax.scatter(
                    p[0, 0],
                    p[0, 1],
                    marker="o",
                    color=color,
                    edgecolors="white",
                    linewidths=0.65,
                    s=33,
                    alpha=0.93,
                    zorder=5,
                )
                ax.scatter(
                    p[-1, 0],
                    p[-1, 1],
                    marker="s",
                    color=color,
                    edgecolors="white",
                    linewidths=0.65,
                    s=33,
                    alpha=0.93,
                    zorder=5,
                )
        ax.set_title("Közös látens tér · valamennyi tesztidősor", pad=14, fontsize=15)
        ax.set_xlabel(f"1. főkomponens ({shared.explained_variance_ratio_[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"2. főkomponens ({shared.explained_variance_ratio_[1]:.1%})", fontsize=12)
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=model_colors[model],
                    lw=3.1,
                    label=f"{model} · {class_counts[model]} trajektória",
                )
                for model in self.experiment.MODEL_NAMES
            ],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            fontsize=11,
        )
        self._panel_style(ax)
        ax.tick_params(labelsize=10)

        # Right-middle horizontal bars: directed Hausdorff distances and P95 in original d_e space.
        side = fig.add_subplot(grid[1, 3:])
        bars = np.array(
            [metrics["directed_a_to_b"], metrics["directed_b_to_a"], metrics["hausdorff_p95"]],
            dtype=float,
        )
        ypos = np.arange(3)
        bars_colors = [PALETTE["teal"], PALETTE["magenta"], PALETTE["blue"]]
        side.barh(ypos, bars, height=0.55, color=bars_colors, alpha=0.92)
        side.set_yticks(ypos, labels=[f"{model_a} → {model_b}", f"{model_b} → {model_a}", "Módosított H95"])
        side.invert_yaxis()
        max_val = float(max(bars.max(), 1e-8))
        side.set_xlim(0, max_val * 1.30)
        for i, v in enumerate(bars):
            side.text(
                v + max_val * 0.025,
                i,
                f"{v:.5g}",
                va="center",
                ha="left",
                color=PALETTE["ink"],
                fontweight="bold",
                fontsize=10,
            )
        side.set_xlabel(f"Távolság az eredeti {embeddings.shape[1]}-dimenziós térben")
        side.set_title("Irányított Hausdorff-értékek", fontsize=12, pad=9)
        side.grid(axis="x")
        side.set_axisbelow(True)
        side.spines["left"].set_visible(False)
        side.tick_params(axis="y", length=0, labelsize=10)

        # Top-right headline readout: symmetric Hausdorff distance in original d_e space.
        headline = fig.add_subplot(grid[0, 3:])
        headline.axis("off")
        headline.text(0.5, 0.75, "Hausdorff-távolság", ha="center", va="center", color=PALETTE["muted"], fontsize=12)
        headline.text(
            0.5,
            0.27,
            f"{metrics['hausdorff']:.6g}",
            ha="center",
            va="center",
            color=PALETTE["ink"],
            fontsize=30,
            fontweight="bold",
        )

        # Bottom-right summary block: L2-normalized Hausdorff, projected variance, and sample count.
        note = fig.add_subplot(grid[2, 3:])
        note.axis("off")
        note.text(0, 0.99, "Kiegészítő mértékek", color=PALETTE["ink"], fontsize=12, fontweight="semibold", va="top")
        note.text(
            0,
            0.75,
            f"Egységvektor-normalizált H:  {metrics['hausdorff_unit_normalized']:.5g}\n"
            f"Normalizált H95:  {metrics['hausdorff_p95_unit_normalized']:.5g}\n"
            f"PCA-vetület varianciája:  {sum(shared.explained_variance_ratio_):.1%}\n"
            f"Tesztpontok száma:  {len(embeddings)}",
            va="top",
            fontsize=10.7,
            linespacing=1.55,
            color=PALETTE["muted"],
        )
        fig.text(
            0.5,
            0.035,
            "A középső ábra 2D PCA-vetület; a Hausdorff-metrikákat "
            "az eredeti beágyazási térben számoltuk. "
            "A színek szerinti elkülönülés önmagában nem bizonyít "
            "címkefüggetlen modellazonosítást.",
            ha="center",
            va="bottom",
            fontsize=9.6,
            color=PALETTE["muted"],
        )

        destination = os.path.join(
            self.output_dir,
            f"multimodel_kozos_latens_ter_hausdorff_{model_a.lower()}_"
            f"{model_b.lower()}_de_{self.experiment.EMBEDDING_DIM}.png",
        )
        self._save_plot(fig, destination)

        # Persist raw embeddings, labels, PCA basis, and summary scalars for downstream analysis.
        np.savez_compressed(
            os.path.splitext(destination)[0] + ".npz",
            feature_embeddings=embeddings,
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
