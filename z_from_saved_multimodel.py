"""Közös Feature-látens tér újrarajzolása a mentett tesztbeágyazásokból.

KIZÁRÓLAG az aktív LaTeX-módszertanban szereplő elemzések:
  - egy közös PCA az ÖSSZES teszttrajektória összes időpontjára;
  - Silhouette-score a valódi modellcímkék alapján, az EREDETI d_e térben.

NINCS tanítás, inferencia, modellbetöltés, tesztadat-generálás,
4x4-es ábra, Procrustes, koszinusz, Hausdorff-számítás stb.

Futtatás a projekt gyökérkönyvtárából:
    python einn_kozos_latens_ter_silhouette.py

Szükséges: numpy, scikit-learn, matplotlib.
Ha van einn_model_to_id.pt, annak olvasásához PyTorch is szükséges.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# =====================================================================
# BEÁLLÍTÁSOK
# =====================================================================

EXPERIMENTS = [
    {
        "pair": ("SIR", "SIS"),
        "run_dir": "./outputs/sir_sisEZAZUJZOLI_D_E_5/label_conditioned/raw",
    },
    {
        "pair": ("SIR", "SEIR"),
        "run_dir": "./outputs/sir_seirEZAZUJZOLI_D_E_5/label_conditioned/raw",
    },
]

# Ha nem létezik a modell-ID mentés, az eredeti MODEL_NAMES sorrendből indulunk ki.
# Különleges címkézésnél itt adható meg kézzel a tényleges párosítás:
# pl. {"SIR": 1, "SIS": 0}. Egyébként hagyd None-on.
MODEL_TO_ID_OVERRIDE = None

POINT_SIZE = 30           # scatter-pontok TERÜLETE pont^2-ben
POINT_ALPHA = 0.46
CURVE_WIDTH = 1.4
CURVE_ALPHA = 0.42
ENDPOINT_SIZE = 55
FIGSIZE = (15.5, 10.0)
PNG_DPI = 260

# Külön almappa: semmilyen régi kimenetet nem írunk felül.
OUTPUT_SUBDIR = "ujrarajzolt_kozos_latens_ter"

PALETTE = {
    "ink": "#18283D",
    "muted": "#607185",
    "grid": "#D8E1EB",
    "teal": "#008E97",
    "magenta": "#AB4C77",
}


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
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


def read_saved_embeddings(run_dir: Path):
    """Csak a mentett tesztbeágyazásokat tölti be, tanult súlyokat nem."""
    candidates = (
        run_dir / "all_heldout_feature_embeddings.npz",
        run_dir / "feature_embeddings_test.npz",
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Nem található mentett tesztbeágyazás: {run_dir}\n"
            "Szükséges: all_heldout_feature_embeddings.npz vagy feature_embeddings_test.npz"
        )
    with np.load(path, allow_pickle=False) as saved:
        required = {"feature_embeddings", "true_labels", "trajectory_ids", "time_indices"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Hiányzó NPZ-mezők ({path.name}): {sorted(missing)}")
        E = np.asarray(saved["feature_embeddings"], dtype=np.float64)
        labels = np.asarray(saved["true_labels"], dtype=int).reshape(-1)
        trajectory_ids = np.asarray(saved["trajectory_ids"], dtype=int).reshape(-1)
        time_indices = np.asarray(saved["time_indices"], dtype=int).reshape(-1)
    if E.ndim != 2 or E.shape[1] < 2 or E.shape[0] < 3:
        raise ValueError(f"Nem megfelelő embedding-mátrix: {E.shape}")
    if any(len(v) != len(E) for v in (labels, trajectory_ids, time_indices)):
        raise ValueError("Az NPZ mezői különböző számú időpontot tartalmaznak.")
    if not np.isfinite(E).all():
        raise ValueError("A mentett beágyazások NaN/Inf értékeket tartalmaznak.")
    return path, E, labels, trajectory_ids, time_indices


def resolve_model_ids(run_dir: Path, pair: tuple[str, str], labels: np.ndarray):
    if MODEL_TO_ID_OVERRIDE is not None:
        mapping = dict(MODEL_TO_ID_OVERRIDE)
    else:
        mapping_path = run_dir / "einn_model_to_id.pt"
        if mapping_path.exists():
            try:
                import torch  # csak a címkék opcionális betöltéséhez szükséges
                saved = torch.load(mapping_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Nem tudtam beolvasni a modellcímkéket: {mapping_path}. "
                    "Add meg a MODEL_TO_ID_OVERRIDE változóban a helyes címkéket."
                ) from exc
            if not isinstance(saved, dict):
                raise TypeError("A modellcímkék mentése nem szótár.")
            mapping = {str(k).upper(): int(v) for k, v in saved.items()}
        else:
            mapping = {pair[0]: 0, pair[1]: 1}
            print("  Figyelem: nincs mentett modell-ID fájl, az eredeti sorrendet feltételezem:", mapping)
    if not all(name in mapping for name in pair):
        raise ValueError(f"A modellcímke-fájl nem tartalmazza a kért modelleket: {pair}, {mapping}")
    if mapping[pair[0]] == mapping[pair[1]]:
        raise ValueError("A két modell nem rendelkezhet azonos címkével.")
    expected = {mapping[name] for name in pair}
    found = set(np.unique(labels).tolist())
    if found != expected:
        raise ValueError(f"Az NPZ címkéi ({found}) nem egyeznek a modell-ID-kkel ({mapping}).")
    return mapping


def calculate_silhouette(E: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette-score: időpontszintű Feature-beágyazások az eredeti d_e térben.

    A PCA kizárólag az ábrázoláshoz kell, a mérőszámba nem kerül bele.
    A sklearn számítása szükség szerint blokkokban végzi a távolságszámítást.
    """
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or any(count < 2 for count in counts):
        raise ValueError(
            "A silhouette-score-hoz két valódi modellcímke és "
            "modellcímkénként legalább két beágyazási pont szükséges."
        )
    return float(silhouette_score(E, labels, metric="euclidean"))


def plot_joint_space(run_dir: Path, pair: tuple[str, str]):
    path, E, labels, trajectory_ids, time_indices = read_saved_embeddings(run_dir)
    model_ids = resolve_model_ids(run_dir, pair, labels)
    silhouette = calculate_silhouette(E, labels)

    # Egy PCA, egyetlen összevont adathalmazon — nem külön görbénként!
    fitted = PCA(n_components=2, svd_solver="full")
    coords = fitted.fit_transform(E)
    variance = fitted.explained_variance_ratio_
    d_e = E.shape[1]

    configure_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=.095, right=.97, bottom=.10, top=.80)

    fig.suptitle(
        f"EINN · közös látens tér   |   {pair[0]} / {pair[1]}   |   $d_e={d_e}$",
        fontsize=19, fontweight="bold", color=PALETTE["ink"], y=.965,
    )
    fig.text(
        .5, .912,
        "Az összes teszttrajektória · egyetlen közös PCA-bázis · pontok időrendben összekötve",
        fontsize=11.4, color=PALETTE["muted"], ha="center",
    )
    fig.text(
        .5, .864,
        rf"Silhouette-score: $S={silhouette:.4f}$",
        fontsize=20.6, color=PALETTE["ink"], ha="center",
    )

    colors = {pair[0]: PALETTE["teal"], pair[1]: PALETTE["magenta"]}
    counts = {}
    for name in pair:
        class_id = model_ids[name]
        class_trajectory_ids = np.unique(trajectory_ids[labels == class_id])
        counts[name] = len(class_trajectory_ids)
        for tid in class_trajectory_ids:
            indices = np.flatnonzero((trajectory_ids == tid) & (labels == class_id))
            # Kritikus: minden eredeti idősort saját sorrendjében kell összekötni.
            indices = indices[np.argsort(time_indices[indices], kind="stable")]
            p = coords[indices]
            ax.plot(p[:, 0], p[:, 1], color=colors[name], lw=CURVE_WIDTH,
                    alpha=CURVE_ALPHA, zorder=2)
            ax.scatter(p[:, 0], p[:, 1], color=colors[name], s=POINT_SIZE,
                       alpha=POINT_ALPHA, edgecolors="none", zorder=3, rasterized=True)
            ax.scatter(*p[0], marker="s", s=ENDPOINT_SIZE, color=colors[name],
                       edgecolors="white", linewidths=.7, zorder=5)
            ax.scatter(*p[-1], marker="^", s=ENDPOINT_SIZE, color=colors[name],
                       edgecolors="white", linewidths=.7, zorder=5)

    ax.set_title("Valamennyi tesztidősor látens pályája", fontsize=14, pad=12)
    ax.set_xlabel(f"1. főkomponens ({variance[0]:.1%})", fontsize=20)
    ax.set_ylabel(f"2. főkomponens ({variance[1]:.1%})", fontsize=20)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)

    ax.legend(handles=[
        Line2D([0], [0], color=colors[name], lw=2.7, marker="o",
               markersize=6, label=f"{name} · {counts[name]} trajektória")
        for name in pair
    ] + [
        Line2D([0], [0], color=PALETTE["ink"], lw=0,
               marker="s", markersize=7, label="Kezdet"),
        Line2D([0], [0], color=PALETTE["ink"], lw=0,
               marker="^", markersize=7, label="Végpont"),
    ], loc="best", frameon=True, framealpha=.94, fontsize=18)

    fig.text(.5, .031,
             "A silhouette-score-t az eredeti $d_e$-dimenziós térben számítottuk, nem a PCA-vetületben.",
             ha="center", color=PALETTE["muted"], fontsize=10)

    out_dir = run_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"kozos_latens_ter_silhouette_{pair[0].lower()}_{pair[1].lower()}_de_{d_e}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Egyetlen könnyen átvehető számszerű táblázat; nem készül másik ábra.
    with (out_dir / f"{stem}_meroszamok.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_a", "model_b", "d_e", "silhouette_score",
                         "pc1_variance", "pc2_variance", "n_test_points",
                         "n_trajectories_a", "n_trajectories_b"])
        writer.writerow([pair[0], pair[1], d_e, silhouette,
                         variance[0], variance[1], len(E), counts[pair[0]], counts[pair[1]]])

    print(f"  Forrás: {path.name} | {len(E)} időpont | d_e={d_e}")
    print(f"  Silhouette-score (eredeti d_e tér): {silhouette:.6f}")
    print(f"  Kész: {png}")
    print(f"        {pdf}")


def main():
    for exp in EXPERIMENTS:
        run_dir = Path(exp["run_dir"]).expanduser()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Nem található eredménykönyvtár: {run_dir.resolve()}")
        print(f"\nModellpár: {exp['pair']} | könyvtár: {run_dir.resolve()}")
        plot_joint_space(run_dir, tuple(exp["pair"]))
    print("\nMindkét modellpár közös látens tere újrarajzolva. Újratanítás nem történt.")


if __name__ == "__main__":
    main()
