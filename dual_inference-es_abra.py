"""SIR-kísérlet: a MENTETT EINN-modulokból készíti újra a 4×4-es ábrát.

Nincs tanítás, nincs checkpoint-átírás és nincs 2×2-es ábrakészítés.
Futtatás a projekt gyökeréből:
    python redraw_sir_dual_inference_4x4.py
    python redraw_sir_dual_inference_4x4.py --font-scale 1.25 --device cuda:1

Elvárt fájlok az experiment_results/sir_dual_inference mappában:
    einn_time_module.pth, einn_feature_module.pth, einn_output_module.pth,
    scaling_stats.pt, 50_states_100len_target_1.pt.

MEGJEGYZÉS: a mentett .pth súlyok nem tartalmazzák az eredeti, zajos
szintetikus TESZT-bemeneteket vagy az akkori RNG-állapotot. Azonos paraméterek
mellett új tesztadatot generálunk. Ezért az ábra MSE-i és predikciói
kismértékben eltérhetnek a korábbi PDF-től. A tanult modell változatlan.
"""

import os
import argparse
from pathlib import Path
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Ábramentés grafikus felület nélkül is.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.sir_model import SIRModel
from einn.inference.einn_evaluator import EINNEvaluator
from einn.utils.experiment_utils import ExperimentUtils
from einn.model.einn_builder import EINNBuilder
from einn.data.data_preprocessor import DataPreprocessor

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
TARGET_IDX = 1  # SIR modellnél az Infected (I) indexe az 1-es
NUM_STATES = 50
SEQ_LEN = 100
DEVICE = 'cuda:1'  # Ha elérhető; ellenkező esetben CPU-ra váltunk.
OUTPUT_DIR = "./experiment_results/sir_dual_inference"
FONT_MULTIPLIER = 1.0  # Parancssorból is állítható: --font-scale 1.2


def fs(size):
    """A teljes, 4×4-es ábra egységes betűméret-szorzója."""
    return round(size * FONT_MULTIPLIER, 1)



# ======================================================================
# KÖZÖS ÁBRASTÍLUS – egyezik az EINN általánosításvizsgálat ábráival
# ======================================================================
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


def configure_plot_style() -> None:
    """A 4×4-es publikációs ábra nagy, egységes betűméretei."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": fs(21),
        "axes.titlesize": fs(25),
        "axes.titleweight": "semibold",
        "axes.labelsize": fs(22),
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
        "xtick.labelsize": fs(20),
        "ytick.labelsize": fs(20),
        "legend.frameon": False,
    })


def _style_panel(ax, time_axis: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", labelsize=fs(20), length=5)
    if time_axis:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def _finite_limits(values, relative_pad: float = 0.10, min_span: float = 0.01):
    """Egységes tengelyhatárok azonos PCA-vetítést ábrázoló sorokhoz."""
    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return -1.0, 1.0
    lo, hi = float(data.min()), float(data.max())
    span = max(hi - lo, min_span)
    return lo - span * relative_pad, hi + span * relative_pad


def _plot_pca_pair(ax, coords_a, coords_b, t_end: int, title: str,
                   xlim, ylim):
    """Két látens pálya AZONOS PCA-térben, a négy sorban azonos tengelyekkel."""
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
        # Ugyanazok a kezdet/vég/megfigyelés-vége jelölések, mint a 4x4-es ábrán.
        ax.scatter(*coords[0], marker="x", color=PALETTE["ink"],
                   s=60, linewidths=1.75, zorder=5)
        ax.scatter(*coords[-1], marker="s", color=PALETTE["orange"],
                   s=61, zorder=6)
        ax.scatter(*coords[t_end-1], marker="o", facecolors="none",
                   edgecolors=PALETTE["magenta"], s=120,
                   linewidths=2.1, zorder=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=12)
    ax.set_xlabel("1. főkomponens")
    ax.set_ylabel("2. főkomponens")
    _style_panel(ax)


def plot_dual_inference_results(phase_results, true_target_1, true_target_2,
                                output_dir, embedding_dim: int):
    """Az eredeti 4×4-es elrendezés, változatlan adatmegjelenítéssel."""
    if len(phase_results) != 4:
        raise ValueError("A 4×4 elrendezéshez pontosan négy megfigyelési ablak kell.")

    configure_plot_style()
    target_a = np.asarray(true_target_1, dtype=float).reshape(-1)
    target_b = np.asarray(true_target_2, dtype=float).reshape(-1)
    if target_a.shape != target_b.shape:
        raise ValueError("A két referencia-idősor hossza nem egyezik.")
    horizon = len(target_a)
    times = np.arange(horizon)

    pca_keys = (
        ("pca_common_1", "pca_common_2", "Közös PCA-vetítés"),
        ("pca_a_1", "pca_a_2", "A trajektória\nPCA-bázisa"),
        ("pca_b_1", "pca_b_2", "B trajektória\nPCA-bázisa"),
    )
    # Mind a négy sorban azonos skála, de a három különböző PCA-bázisban
    # szükség esetén eltérhet a koordinátatartomány.
    pca_limits = []
    for a_key, b_key, _ in pca_keys:
        combined = np.concatenate([
            np.asarray(phase[key], dtype=float)
            for phase in phase_results
            for key in (a_key, b_key)
        ], axis=0)
        pca_limits.append((
            _finite_limits(combined[:, 0], min_span=.02),
            _finite_limits(combined[:, 1], min_span=.02),
        ))

    predictions = [
        np.asarray(phase[f"results_{i}"]["predicted_trajectory"], dtype=float).reshape(-1)
        for phase in phase_results for i in (1, 2)
    ]
    if any(p.size != horizon for p in predictions):
        raise ValueError("Az egyik becsült trajektória hossza eltér a referenciától.")
    pred_limits = _finite_limits(
        np.concatenate([target_a, target_b, *predictions]),
        relative_pad=.08, min_span=.015,
    )

    fig = plt.figure(figsize=(28.5, 24.8), facecolor="white")
    grid = fig.add_gridspec(
        4, 4, left=.061, right=.987, top=.790, bottom=.057,
        wspace=.48, hspace=.80,
    )
    fig.suptitle(
        "EINN · két eltérő SIR-járványdinamika összehasonlítása"
        f"   |   $d_e={embedding_dim}$",
        fontsize=fs(33), fontweight="bold", color=PALETTE["ink"], y=.988,
    )
    fig.text(
        .5, .952,
        "Négy megfigyelési ablak · két független teszttrajektória · összehasonlítható PCA-tengelyek",
        ha="center", fontsize=fs(23), color=PALETTE["muted"],
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
               bbox_to_anchor=(.5, .922), ncol=5, frameon=False, fontsize=fs(21))
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
               bbox_to_anchor=(.5, .885), ncol=3, frameon=False, fontsize=fs(21))

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
        ax.set_title(f"{end} megfigyelt időpont\njárványdinamika", pad=12)
        ax.set_xlabel("Időindex")
        ax.set_ylabel("Fertőzöttek aránya")
        ax.text(
            .97, .96,
            f"Jövőbeli MSE  A: {float(r_a['mse_future']):.3g}\n"
            f"Jövőbeli MSE  B: {float(r_b['mse_future']):.3g}",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=fs(17), color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=.34", facecolor="white",
                      edgecolor=PALETTE["grid"], alpha=.93),
            zorder=10,
        )
        _style_panel(ax, time_axis=True)

        for j, (a_key, b_key, title) in enumerate(pca_keys, start=1):
            ax = fig.add_subplot(grid[row, j])
            xlim, ylim = pca_limits[j-1]
            _plot_pca_pair(ax, phase[a_key], phase[b_key], end,
                           title, xlim, ylim)

    filename = os.path.join(output_dir, "sir_dual_inference_pca_comparison_large_fonts.png")
    fig.savefig(filename, dpi=250, bbox_inches="tight", pad_inches=.30)
    pdf_filename = os.path.splitext(filename)[0] + ".pdf"
    fig.savefig(pdf_filename, bbox_inches="tight", pad_inches=.30)
    plt.close(fig)
    return filename, pdf_filename




def _get_device(requested: str) -> str:
    """Ellenőrzi a kért CUDA-eszközt, illetve automatikus választást ad."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        return "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"A kért eszköz ({requested}) nem érhető el: nincs CUDA.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Nincs {requested}; elérhető CUDA-eszközök: {torch.cuda.device_count()}.")
    return requested


def _load_existing(path, device="cpu"):
    """Csak megbízható, helyben mentett PyTorch-fájlokhoz."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Hiányzó mentett fájl: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def regenerate_plot_from_saved_weights(output_dir: str, device: str) -> None:
    """Tesztszimuláció -> betöltött hálózat -> inferencia -> PCA -> eredeti 4×4."""
    ExperimentUtils.seed_everything(seed=42)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    required = (
        "einn_time_module.pth", "einn_feature_module.pth",
        "einn_output_module.pth", "scaling_stats.pt",
        f"{NUM_STATES}_states_{SEQ_LEN}len_target_{TARGET_IDX}.pt",
    )
    missing = [f for f in required if not os.path.isfile(os.path.join(output_dir, f))]
    if missing:
        raise FileNotFoundError(
            "A következő mentett fájlok hiányoznak a mappából: " + ", ".join(missing)
            + f"\nKeresett mappa: {output_dir}"
        )

    print("1. Mentett skálázási adatok és cache beolvasása (nincs tanítás)")
    scaling_stats = _load_existing(os.path.join(output_dir, "scaling_stats.pt"))
    cached_train_data = _load_existing(
        os.path.join(output_dir, f"{NUM_STATES}_states_{SEQ_LEN}len_target_{TARGET_IDX}.pt")
    )
    if "x_train" not in cached_train_data:
        raise KeyError("A mentett tanítócache-ben nem található 'x_train'.")
    preprocessor = DataPreprocessor()
    # Pontosan az eredeti ábrakód módszerét követjük: a már mentett x_train
    # adatokból állítjuk vissza az inferenciához alkalmazott x-skálázást.
    preprocessor.x_scaler.fit(np.concatenate(cached_train_data["x_train"]))
    x_scaler = preprocessor.x_scaler

    print("2. Két teszttrajektória előállítása az EREDETI parametrizálással")
    test_correlations = {
        "mobility": -0.75, "temperature": -0.1,
        "testing": 0.85, "stringency": 0.95,
    }
    true_model = SIRModel(target_idx=TARGET_IDX)
    generator_test = SyntheticDataGenerator(true_ode_model=true_model, seq_len=SEQ_LEN)
    data_test_1 = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0], true_params=[0.99, 0.1],
        noise_std=0.01, correlations=test_correlations,
    )
    data_test_2 = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0], true_params=[0.01, 0.1],
        noise_std=0.01, correlations=test_correlations,
    )
    x_test_scaled_1 = x_scaler.transform(data_test_1["x_raw"])
    x_test_scaled_2 = x_scaler.transform(data_test_2["x_raw"])
    true_target_1 = data_test_1["true_latent_states"][:, TARGET_IDX]
    true_target_2 = data_test_2["true_latent_states"][:, TARGET_IDX]
    full_t_raw = torch.arange(SEQ_LEN, dtype=torch.float32).reshape(1, SEQ_LEN, 1).to(device)

    print("3. EINN-architektúra létrehozása és a meglévő .pth súlyok betöltése")
    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=3, d_p=2, feature_n_layers=2)
    # Csak a Builder konfigurációjához szükséges, nem indít trainert.
    train_cfg = EINNTrainConfig(
        device=device, learning_rate=0.001, early_stopping_patience=5,
        loss_weights={
            "data_T": 10.0, "data_F": 10.0, "aux": 4.0,
            "ode_T": 8.0, "ode_F": 8.0,
            "ode_future_T": 0.1, "ode_future_F": 0.1,
            "mono": 0.0, "param": 0.1,
            "kd_target": 10.0, "kd_emb": 10.0,
            "ic": 20.0, "ic_F": 20.0,
        },
        reps={"phase_1": 50, "phase_2": 50, "phase_3": 50, "phase_4": 50},
    )
    models = EINNBuilder.build_einn(model_cfg, train_cfg, {}, "SIR", seed=42)
    for attribute, filename in (
        ("time_module", "einn_time_module.pth"),
        ("feature_module", "einn_feature_module.pth"),
        ("output_module", "einn_output_module.pth"),
    ):
        module = getattr(models, attribute)
        module.load_state_dict(_load_existing(os.path.join(output_dir, filename), device))
        module.to(device)
        module.eval()
        print(f"   Betöltve: {filename}")
    models.ode_model = SIRModel(target_idx=TARGET_IDX)
    evaluator = EINNEvaluator(models=models, device=device)

    print("4. Inferencia a négy megfigyelési ablakra")
    phase_results = []
    for t_end in (5, 10, 25, 50):
        results_1 = evaluator.evaluate(
            x_test=x_test_scaled_1, future_t=full_t_raw,
            scaling_stats=scaling_stats, target_idx=TARGET_IDX,
            true_target_trajectory=true_target_1, t_end=t_end,
        )
        results_2 = evaluator.evaluate(
            x_test=x_test_scaled_2, future_t=full_t_raw,
            scaling_stats=scaling_stats, target_idx=TARGET_IDX,
            true_target_trajectory=true_target_2, t_end=t_end,
        )
        phase_results.append({
            "phase_name": f"{t_end} időpont", "t_end": t_end,
            "results_1": results_1, "results_2": results_2,
        })
        print(f"   t_end={t_end}: MSE A={float(results_1['mse_future']):.6g}, "
              f"MSE B={float(results_2['mse_future']):.6g}")

    print("5. Az eredeti három PCA-bázis újraszámítása (a mentett hálózat kimenetein)")
    all_embeddings_a = np.concatenate(
        [phase["results_1"]["latent_feature"] for phase in phase_results], axis=0)
    all_embeddings_b = np.concatenate(
        [phase["results_2"]["latent_feature"] for phase in phase_results], axis=0)
    all_embeddings = np.concatenate([all_embeddings_a, all_embeddings_b], axis=0)
    pca_common = PCA(n_components=2).fit(all_embeddings)
    pca_a = PCA(n_components=2).fit(all_embeddings_a)
    pca_b = PCA(n_components=2).fit(all_embeddings_b)
    for phase in phase_results:
        emb_a = phase["results_1"]["latent_feature"]
        emb_b = phase["results_2"]["latent_feature"]
        phase["pca_common_1"] = pca_common.transform(emb_a)
        phase["pca_common_2"] = pca_common.transform(emb_b)
        phase["pca_a_1"] = pca_a.transform(emb_a)
        phase["pca_a_2"] = pca_a.transform(emb_b)
        phase["pca_b_1"] = pca_b.transform(emb_a)
        phase["pca_b_2"] = pca_b.transform(emb_b)

    print("6. EGYETLEN 4×4-es ábra mentése, nagyobb betűkkel")
    png_path, pdf_path = plot_dual_inference_results(
        phase_results=phase_results,
        true_target_1=true_target_1,
        true_target_2=true_target_2,
        output_dir=output_dir,
        embedding_dim=model_cfg.d_e,
    )
    print(f"Kész: {pdf_path}\n      {png_path}")


def main():
    global FONT_MULTIPLIER
    parser = argparse.ArgumentParser(description="SIR: 4×4-es PDF újragenerálása .pth-ból, tanítás nélkül")
    parser.add_argument(
        "--checkpoint-dir", default=OUTPUT_DIR,
        help="A mentett .pth/.pt állományokat tartalmazó könyvtár",
    )
    parser.add_argument(
        "--device", default="auto", help="auto | cpu | cuda:0 | cuda:1",
    )
    parser.add_argument(
        "--font-scale", default=1.0, type=float,
        help="Betűméret-szorzó (1.0=alap nagy; 1.2=még nagyobb)",
    )
    args = parser.parse_args()
    if not 0.5 <= args.font_scale <= 1.8:
        parser.error("A --font-scale értéke 0.5 és 1.8 között legyen.")
    FONT_MULTIPLIER = args.font_scale
    device = _get_device(args.device)
    print(f"Eszköz: {device}")
    regenerate_plot_from_saved_weights(args.checkpoint_dir, device)


if __name__ == "__main__":
    main()