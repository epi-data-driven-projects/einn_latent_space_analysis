#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EINN egyszeres modell: kizárólag a korábban elmentett adatok ÚJRARAJZOLÁSA.

NEM tanít hálózatot, NEM épít modellt, és NEM generál teszttrajektóriát.

Példák a projekt gyökérkönyvtárából:
    python einn_abrak_ujrarajzolasa_4x4.py
    python einn_abrak_ujrarajzolasa_4x4.py --model SIS --dim 20
    python einn_abrak_ujrarajzolasa_4x4.py --models SIR SIS ROST --dims 20 5
    python einn_abrak_ujrarajzolasa_4x4.py --cosine-range 35 110
    python einn_abrak_ujrarajzolasa_4x4.py --root /masik/eredmeny/mappa

Bemenet, minden modell/d_e mappában (legalább az egyik):
  A) test_latent_representations.npz   (befejezett futás)
  B) ablak_<N>_ideiglenes_eredmenyek.npz, mind a négy megfigyelési ablakra

Kimenet, ugyanabba a mappába, csak ÚJ fájlnevekkel:
- ..._4x4.png és ..._4x4.pdf
- ..._koszinusz_nagyitva.png/.pdf
- ..._procrustes.png/.pdf
- uj_abrak_numerikus_osszefoglalo.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter
import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

# ------ Ábrázolási beállítások ------
DEFAULT_ROOT = Path("./experiment_results/generalization")
# Pontosan a korábbi RUN_ALL_MODELS=True kísérlet hat futtatása:
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

# A fő ábra színeitől (teal, kék, narancs, magenta) teljesen eltérő paletta a Procrustes sávokhoz
PROCRUSTES_COLORS = (
    "#3B7A57",  # erdőzöld
    "#6B4E9B",  # mélylila
    "#C44536",  # tompa cinóbervörös
    "#8C6D46",  # meleg bronz
)


@dataclass
class Experiment:
    name: str
    dimension: int
    observations: np.ndarray              # [4]
    target: np.ndarray                    # [T]
    predictions: np.ndarray               # [4,T]
    time_embeddings: np.ndarray           # [4,T,d_e]
    feature_embeddings: np.ndarray        # [4,T,d_e]
    cosine: np.ndarray                    # [4,T]
    pca_coords: np.ndarray                # [4,T,2]
    cumulative_variance: np.ndarray       # [4,d_e]
    shared_variance_ratio: np.ndarray     # [2]
    disparity: np.ndarray                 # [4]
    mse_future: np.ndarray                # [4]
    d95: np.ndarray                       # [4]
    d99: np.ndarray                       # [4]
    source: str


def style() -> None:
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


def _choose_file(directory: Path) -> tuple[str, Path | list[Path]]:
    full = directory / "test_latent_representations.npz"
    if full.is_file():
        return "complete", full
    parts = sorted(directory.glob("ablak_*_ideiglenes_eredmenyek.npz"))
    if parts:
        return "temporary", parts
    raise FileNotFoundError(
        f"Hiányoznak a mentett kiértékelési fájlok: {directory}\n"
        "Várt bemenet: test_latent_representations.npz vagy "
        "ablak_*_ideiglenes_eredmenyek.npz."
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.einsum("td,td->t", a, b)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numerator, denominator,
        out=np.full(a.shape[0], np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )


def _procrustes(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - np.mean(a, axis=0)
    b0 = b - np.mean(b, axis=0)
    if np.linalg.norm(a0) <= 1e-12 or np.linalg.norm(b0) <= 1e-12:
        return float("nan")
    _, _, disparity = procrustes(a, b)
    return float(disparity)


def _load_completed(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        required = (
            "observation_windows", "true_target", "predictions",
            "time_embeddings", "feature_embeddings",
        )
        missing = [key for key in required if key not in saved]
        if missing:
            raise ValueError(f"{path}: hiányzó kulcs(ok): {missing}")
        result = {key: np.array(saved[key], copy=True) for key in required}
        if "cosine_similarities" in saved:
            result["cosine"] = np.array(saved["cosine_similarities"], copy=True)
    return result


def _load_temporary(files: Sequence[Path]):
    results = []
    target = None
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            required = ("observed_steps", "true_target", "prediction", "time_embedding", "feature_embedding")
            missing = [key for key in required if key not in saved]
            if missing:
                raise ValueError(f"{path}: hiányzó kulcs(ok): {missing}")
            obs = int(np.asarray(saved["observed_steps"]).item())
            current_target = np.asarray(saved["true_target"], dtype=np.float64).reshape(-1).copy()
            if target is None:
                target = current_target
            elif not np.allclose(target, current_target, rtol=1e-8, atol=1e-10):
                raise ValueError("A mentett ablakok eltérő teszttrajektóriákat tartalmaznak.")
            entry = {
                "obs": obs,
                "pred": np.asarray(saved["prediction"], dtype=np.float64).reshape(-1).copy(),
                "time": np.array(saved["time_embedding"], copy=True),
                "feature": np.array(saved["feature_embedding"], copy=True),
            }
            if "cosine_similarity" in saved:
                entry["cosine"] = np.asarray(saved["cosine_similarity"], dtype=np.float64).reshape(-1).copy()
            results.append(entry)
    results.sort(key=lambda x: x["obs"])
    observed = [item["obs"] for item in results]
    if len(set(observed)) != len(observed):
        raise ValueError(f"Ismétlődő megfigyelési ablakok: {observed}")
    if len(results) != 4:
        raise ValueError(f"Négy megfigyelési ablak kell, jelenleg {len(results)}: {observed}")
    result = {
        "observation_windows": np.asarray(observed, dtype=int),
        "true_target": target,
        "predictions": np.stack([x["pred"] for x in results]),
        "time_embeddings": np.stack([x["time"] for x in results]),
        "feature_embeddings": np.stack([x["feature"] for x in results]),
    }
    if all("cosine" in x for x in results):
        result["cosine"] = np.stack([x["cosine"] for x in results])
    return result


def load_experiment(directory: Path, model: str, dimension: int) -> Experiment:
    kind, source = _choose_file(directory)
    arrays = _load_completed(source) if kind == "complete" else _load_temporary(source)
    observed = np.asarray(arrays["observation_windows"], dtype=int).reshape(-1)
    target = np.asarray(arrays["true_target"], dtype=np.float64).reshape(-1)
    predictions = np.asarray(arrays["predictions"], dtype=np.float64)
    times = np.asarray(arrays["time_embeddings"], dtype=np.float64)
    features = np.asarray(arrays["feature_embeddings"], dtype=np.float64)
    if observed.size != 4 or observed.size != len(set(observed)) or not np.all(np.diff(observed) > 0):
        raise ValueError("A 4 megfigyelési ablaknak szigorúan növekvőnek kell lennie.")
    if times.ndim != 3 or times.shape != features.shape:
        raise ValueError(f"Eltérő Time és Feature beágyazásméretek: {times.shape}, {features.shape}")
    count, horizon, de = times.shape
    if count != 4 or de != dimension or horizon != target.size or predictions.shape != (count, horizon):
        raise ValueError(
            f"Méretellentmondás: time={times.shape}, target={target.shape}, "
            f"pred={predictions.shape}, várt d_e={dimension}"
        )
    if np.any(observed <= 0) or np.any(observed >= horizon):
        raise ValueError("Hibás megfigyelési ablak; 1 <= ablak < T kell.")
    if not (np.isfinite(times).all() and np.isfinite(features).all() and
            np.isfinite(predictions).all() and np.isfinite(target).all()):
        raise ValueError("Nem véges értékek találhatók a mentett adatokban.")

    # Az eredeti kiértékeléshez hasonlóan az összes Feature-időpontból EGY közös PCA.
    shared = PCA(n_components=2)
    flat = features.reshape(-1, de)
    pca_coords = shared.fit_transform(flat).reshape(count, horizon, 2)
    cumulative = np.stack([
        np.cumsum(PCA(n_components=min(horizon, de)).fit(x).explained_variance_ratio_)
        for x in features
    ])
    d95 = np.asarray([np.searchsorted(row, .95) + 1 for row in cumulative])
    d99 = np.asarray([np.searchsorted(row, .99) + 1 for row in cumulative])
    cosine = np.stack([_cosine(a, b) for a, b in zip(times, features)])
    if "cosine" in arrays:
        stored = np.asarray(arrays["cosine"], dtype=np.float64)
        if stored.shape == cosine.shape and not np.allclose(stored, cosine, equal_nan=True, rtol=1e-5, atol=1e-7):
            warnings.warn(
                "Az elmentett koszinuszos hasonlóság nem egyezik a beágyazásokból "
                "újraszámolt értékekkel. A mentett értékeket használjuk.",
                stacklevel=2,
            )
        if stored.shape == cosine.shape:
            cosine = stored

    disparity = np.asarray([_procrustes(a, b) for a, b in zip(times, features)])
    mse_future = np.asarray([
        np.mean((predictions[i, end:] - target[end:]) ** 2)
        for i, end in enumerate(observed)
    ], dtype=float)
    return Experiment(
        name=model, dimension=dimension, observations=observed,
        target=target, predictions=predictions,
        time_embeddings=times, feature_embeddings=features,
        cosine=cosine, pca_coords=pca_coords,
        cumulative_variance=cumulative,
        shared_variance_ratio=shared.explained_variance_ratio_,
        disparity=disparity, mse_future=mse_future,
        d95=d95, d99=d99,
        source=str(source if kind == "complete" else f"{len(source)} ideiglenes NPZ"),
    )


def _finite_limits(values: np.ndarray, *, floor: float | None = None,
                   ceiling: float | None = None, relative_pad: float = 0.12,
                   min_span: float = 0.015) -> tuple[float, float]:
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


def _mark_observed(ax, n: int, horizon: int):
    ax.axvspan(-0.5, n - 0.5, color=PALETTE["background_obs"], alpha=0.70, zorder=0)
    ax.axvline(n - 0.5, color=PALETTE["muted"], ls=(0, (4, 3)), lw=1.8, zorder=4)
    ax.set_xlim(0, horizon - 1)


def _panel(ax, time_axis: bool = False):
    ax.set_axisbelow(True)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", labelsize=16.5, length=5)
    if time_axis:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))


def _pca_axis(ax, cum: np.ndarray, d95: int, d99: int, max_components: int):
    show = min(max_components, cum.size)
    k = np.arange(1, show + 1)
    y = cum[:show]
    ax.plot(k, y, color=PALETTE["blue"], lw=4.0, marker="o", markersize=8.0,
            markerfacecolor="white", markeredgewidth=2.2, zorder=4)
    ax.axhline(.95, color=PALETTE["orange"], ls="--", lw=3.2, zorder=3)
    ax.axhline(.99, color=PALETTE["magenta"], ls=(0, (3, 2)), lw=3.2, zorder=3)
    ylo, yhi = _finite_limits(np.r_[y, .95, .99], floor=0., ceiling=1.005,
                              relative_pad=.06, min_span=.03)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(.65, show + .4)
    ax.set_xticks(k)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100*value:.0f}%"))
    ax.set_title(f"PCA · 95%: {d95} és 99%: {d99} komp.", pad=12)
    ax.set_xlabel("Főkomponensek száma")
    ax.set_ylabel("Kumulatív variancia")
    _panel(ax)


def _cosine_axis(ax, values: np.ndarray, times: np.ndarray, n: int, horizon: int):
    ax.plot(times, values, color=PALETTE["navy"], lw=3.4,
            solid_joinstyle="round", zorder=3)
    _mark_observed(ax, n, horizon)
    ylim = _finite_limits(values, floor=-1., ceiling=1., relative_pad=.09, min_span=.0005)
    ax.set_ylim(*ylim)
    ax.set_title("Time–Feature · koszinuszos has.", pad=12)
    ax.set_xlabel("Időindex")
    ax.set_ylabel("Koszinuszos has.")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    _panel(ax, time_axis=True)
    digits = 5 if (ylim[1] - ylim[0]) < .03 else 3
    ax.text(.98, .04, f"{ylim[0]:.{digits}f} – {ylim[1]:.{digits}f}", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=14.5, color=PALETTE["muted"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=3.0))


def _procrustes_bars(ax, exp: Experiment):
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


def _save(fig, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=PNG_DPI, bbox_inches="tight", pad_inches=.22)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


def plot_grid(exp: Experiment, output_dir: Path, pca_components: int) -> Path:
    style()
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

    ypred = _finite_limits(np.concatenate([exp.target, exp.predictions.ravel()]),
                           relative_pad=.08, min_span=.01)
    for i, obs in enumerate(exp.observations):
        n = int(obs)
        ax = fig.add_subplot(grid[i, 0])
        ax.plot(times, exp.target, color="#171717", lw=3.0, zorder=3)
        ax.plot(times, exp.predictions[i], color=PALETTE["teal"], lw=3.2, zorder=3)
        _mark_observed(ax, n, horizon)
        ax.set_ylim(*ypred)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(f"{n} megfigyelés · MSE: {exp.mse_future[i]:.3g}", pad=12)
        ax.set_xlabel("Időindex")
        ax.set_ylabel("Kumulált fertőzöttek" if exp.name == "ROST" else "Fertőzöttek aránya")
        _panel(ax, time_axis=True)

        ax = fig.add_subplot(grid[i, 1])
        _pca_axis(ax, exp.cumulative_variance[i], int(exp.d95[i]), int(exp.d99[i]), pca_components)

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
        _panel(ax)

        ax = fig.add_subplot(grid[i, 3])
        _cosine_axis(ax, exp.cosine[i], times, n, horizon)

    bottom = fig.add_subplot(grid[4, :])
    _procrustes_bars(bottom, exp)
    destination = output_dir / f"{exp.name.lower()}_altalanositas_de_{exp.dimension}_4x4.png"
    _save(fig, destination)
    return destination


def plot_cosine_detail(exp: Experiment, directory: Path,
                       cosine_range: tuple[int, int] | None) -> Path:
    """Külön, teljes szélességű koszinuszos ábra; tengelyek soronként nagyítva."""
    style()
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
        ax.set_ylim(*_finite_limits(values, floor=-1., ceiling=1., relative_pad=.12, min_span=.0003))
        ax.set_ylabel("Koszinuszos\nhasonlóság")
        ax.set_title(f"{n} megfigyelt időpont", loc="left", pad=10, fontsize=19.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        _panel(ax)
    axs[-1].set_xlabel("Időindex")
    axs[-1].xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    destination = directory / f"{exp.name.lower()}_koszinusz_nagyitva_de_{exp.dimension}.png"
    _save(fig, destination)
    return destination


def plot_procrustes_detail(exp: Experiment, directory: Path) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    fig.subplots_adjust(left=.26, right=.95, top=.76, bottom=.20)
    fig.suptitle(f"Procrustes-összehasonlítás   |   {exp.name}, $d_e={exp.dimension}$",
                 fontsize=26, fontweight="bold", y=.96, color=PALETTE["ink"])
    _procrustes_bars(ax, exp)
    destination = directory / f"{exp.name.lower()}_procrustes_de_{exp.dimension}.png"
    _save(fig, destination)
    return destination


def save_summary_csv(exp: Experiment, directory: Path) -> Path:
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


def find_runs(root: Path, models: list[str] | None, dims: list[int] | None) -> list[tuple[str, int, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Az eredménymappa nem létezik: {root.resolve()}")
    names = [x.upper() for x in models] if models else [x.name.upper() for x in root.iterdir() if x.is_dir()]
    names = [n for n in ("SIR", "SIS", "SEIRM", "ROST") if n in set(names)]
    if not names:
        raise FileNotFoundError(f"Nem találtam modelleket a megadott helyen: {root}")
    runs = []
    for model in names:
        model_dir = root / model.lower()
        dimension_list = dims if dims else [20, 5]
        for d in dimension_list:
            ddir = model_dir / f"de_{d}"
            if not ddir.is_dir():
                if models or dims:
                    warnings.warn(f"Nincs ilyen futtatási mappa, kihagyva: {ddir}", stacklevel=2)
                continue
            try:
                _choose_file(ddir)
            except FileNotFoundError:
                if models or dims:
                    warnings.warn(f"Nincsenek NPZ-fájlok, kihagyva: {ddir}", stacklevel=2)
                continue
            runs.append((model, int(d), ddir))
    if not runs:
        raise FileNotFoundError(
            f"Nem találtam újrarajzolható NPZ eredményeket itt: {root.resolve()}\n"
            "Például: sir/de_20/test_latent_representations.npz"
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="EINN 4x4 ábrák újrarajzolása, újratanítás nélkül")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Az experiment_results/generalization gyökérkönyvtára")
    parser.add_argument("--models", nargs="+", type=str, default=None,
                        help="SIR SIS SEIRM ROST (ha hiányzik, automatikus keresés)")
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="5 és/vagy 20 (ha hiányzik, mindkettő)")
    parser.add_argument("--model", type=str, default=None,
                        help="Egyetlen modell (pl. SIS)")
    parser.add_argument("--dim", type=int, default=None,
                        help="Egyetlen látens dimenzió (pl. 20)")
    parser.add_argument("--pca-max", type=int, default=DEFAULT_PCA_COMPONENTS,
                        help="Legfeljebb hány komponenst mutasson a PCA-görbe (alap: 8)")
    parser.add_argument("--cosine-range", nargs=2, type=int, metavar=("KEZDET", "VEG"),
                        help="Koszinuszos részletábra időintervalluma; VÉG nem tartozik bele")
    args = parser.parse_args()
    if args.pca_max <= 0:
        parser.error("--pca-max pozitív egész legyen")
    models = [args.model] if args.model else args.models
    dims = [args.dim] if args.dim is not None else args.dims
    if RUN_ALL_MODELS and models is None and dims is None:
        models = list(ALL_MODEL_NAMES)
        dims = list(ALL_EMBEDDING_DIMS)
    runs = find_runs(args.root, models, dims)
    if RUN_ALL_MODELS and args.models is None and args.model is None and args.dims is None and args.dim is None:
        expected = {(name, d) for name in ALL_MODEL_NAMES for d in ALL_EMBEDDING_DIMS}
        found = {(name, d) for name, d, _ in runs}
        missing = sorted(expected - found)
        if missing:
            raise FileNotFoundError(
                "A RUN_ALL_MODELS=True kísérletből hiányzó mentett eredmény(ek): "
                + ", ".join(f"{n}/de_{d}" for n, d in missing)
                + ". Ellenőrizd a --root útvonalat, illetve a test_latent_representations.npz "
                  "vagy négy ablak_*_ideiglenes_eredmenyek.npz fájlt."
            )
    print(f"Újrarajzolható futtatások: {len(runs)} (újratanítás NINCS)")
    failed = []
    for i, (model, d, directory) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {model}, d_e={d}: {directory}")
        try:
            exp = load_experiment(directory, model, d)
            target_dir = directory / OUTPUT_SUBDIRECTORY
            target_dir.mkdir(parents=True, exist_ok=True)
            p = plot_grid(exp, target_dir, args.pca_max)
            c = plot_cosine_detail(exp, target_dir, tuple(args.cosine_range) if args.cosine_range else None)
            q = plot_procrustes_detail(exp, target_dir)
            m = save_summary_csv(exp, target_dir)
            for item in (p, p.with_suffix(".pdf"), c, c.with_suffix(".pdf"),
                         q, q.with_suffix(".pdf"), m):
                print(f"  → {item}")
        except Exception as exc:
            failed.append((model, d, str(exc)))
            print(f"  HIBA: {exc}")
    if failed:
        for model, d, err in failed:
            print(f"Sikertelen: {model} d_e={d}: {err}")
        raise SystemExit(1)
    print("Az összes elérhető eredmény újrarajzolva.")


if __name__ == "__main__":
    main()
