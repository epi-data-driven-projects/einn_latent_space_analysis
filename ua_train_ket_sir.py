import os
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
from einn.data.einn_input_loader import EINNInputLoader
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.sir_model import SIRModel
from einn_runner import EINNRunner
from einn.inference.einn_evaluator import EINNEvaluator
from einn.utils.experiment_utils import ExperimentUtils
from einn.model.einn_builder import EINNBuilder
from einn.data.data_preprocessor import DataPreprocessor

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
TARGET_IDX = 1  # SIR modellnél az Infected (I) indexe az 1-es
NUM_STATES = 100
SEQ_LEN = 100
DEVICE = 'cuda:1'
OUTPUT_DIR = "./experiment_results/sir_dual_inference"


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
    """Az előző újrarajzoló programmal azonos betűk, színek és rácsozat."""
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


def _style_panel(ax, time_axis: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", labelsize=9, length=3)
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
    ax.set_title(title, pad=9)
    ax.set_xlabel("1. főkomponens")
    ax.set_ylabel("2. főkomponens")
    _style_panel(ax)


def plot_dual_inference_results(phase_results, true_target_1, true_target_2,
                                output_dir, embedding_dim: int):
    """A korábbi EINN 4×4 ábrasorozatának stílusához igazított ábra."""
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
        ("pca_a_1", "pca_a_2", "A trajektória PCA-bázisa"),
        ("pca_b_1", "pca_b_2", "B trajektória PCA-bázisa"),
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

    fig = plt.figure(figsize=(26.5, 18.2), facecolor="white")
    grid = fig.add_gridspec(
        4, 4, left=.058, right=.986, top=.858, bottom=.055,
        wspace=.34, hspace=.60,
    )
    fig.suptitle(
        "EINN · két eltérő SIR-járványdinamika összehasonlítása"
        f"   |   $d_e={embedding_dim}$",
        fontsize=19, fontweight="bold", color=PALETTE["ink"], y=.986,
    )
    fig.text(
        .5, .952,
        "Négy megfigyelési ablak · két független teszttrajektória · összehasonlítható PCA-tengelyek",
        ha="center", fontsize=11.5, color=PALETTE["muted"],
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
               bbox_to_anchor=(.5, .928), ncol=5, frameon=False, fontsize=10)
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
               bbox_to_anchor=(.5, .897), ncol=3, frameon=False, fontsize=9.5)

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
            fontsize=9, color=PALETTE["ink"],
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

    filename = os.path.join(output_dir, "sir_dual_inference_pca_comparison.png")
    fig.savefig(filename, dpi=250, bbox_inches="tight", pad_inches=.18)
    pdf_filename = os.path.splitext(filename)[0] + ".pdf"
    fig.savefig(pdf_filename, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    return filename, pdf_filename


def run_sir_dual_inference_experiment():
    """
    Executes the full pipeline for the SIR Epidemic Model, then evaluates
    two differently parameterized SIR trajectories side-by-side on a 4x4 plot.
    """
    ExperimentUtils.seed_everything(seed=42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- 1. DATA PREPARATION ---")
    loader = EINNInputLoader(
        true_model=SIRModel(target_idx=TARGET_IDX),
        calibrator_model=SIRModel(target_idx=TARGET_IDX),  # <-- EZ A SOR HIÁNYZOTT
        target_idx=TARGET_IDX,
        device=DEVICE,
        cache_dir=OUTPUT_DIR
    )

    # SIR fizikai paraméter tartományok (beta, gamma)
    SIR_PARAM_RANGES = {
        "beta": (0, 1.0),
        "gamma": (0, 1.0)
    }

    dataset, scaling_stats = loader.prepare_training_data(
        num_states=NUM_STATES, 
        seq_len=SEQ_LEN, 
        d_p=2, 
        param_ranges=SIR_PARAM_RANGES,
        initial_states=[0.99, 0.01, 0.0],
        noise_std=0.01,
        window_step=40 # Gyorsabb lefutás kísérletezéshez
    )

    print("\n--- 2. MODEL TRAINING ---")
    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=3, d_p=2, feature_n_layers=2)
    
    train_cfg = EINNTrainConfig(
        device=DEVICE, learning_rate=0.001, early_stopping_patience=5,
        loss_weights={
            'data_T': 10.0, 'data_F': 10.0,  
            'aux': 4.0,
            'ode_T': 8.0, 'ode_F': 8.0,
            'ode_future_T': 0.1, 'ode_future_F': 0.1,
            'mono': 0.0,
            'param': 0.1,
            'kd_target': 10.0, 'kd_emb': 10.0,
            'ic': 20.0, 'ic_F': 20.0
        },
        reps={'phase_1': 50, 'phase_2': 50, 'phase_3': 50, 'phase_4': 50}
    )

    sir_calibration = {"beta": 0.4, "gamma": 0.1}

    runner = EINNRunner(output_dir=OUTPUT_DIR, model_cfg=model_cfg, train_cfg=train_cfg, seed=42)
    runner.train_and_save(
        dataset=dataset,
        scaling_stats=scaling_stats,
        model_type="SIR",
        target_idx=TARGET_IDX,
        param_calibration=sir_calibration,
        epochs=10,
        batch_size=32
    )

    # ==========================================
    # 3. GENERATE UNSEEN TEST DATA (2 DIFFERENT TRAJECTORIES)
    # ==========================================
    print("\n--- 3. GENERATING UNSEEN TEST STATES ---")
    test_correlations = {
        'mobility': -0.75, 'temperature': -0.1, 'testing': 0.85, 'stringency': 0.95
    }
    
    true_model = SIRModel(target_idx=TARGET_IDX)
    generator_test = SyntheticDataGenerator(true_ode_model=true_model, seq_len=SEQ_LEN)
    
    # 1. Trajektória: Magas fertőzőképesség, lassú gyógyulás (Nagy haranggörbe)
    data_test_1 = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0],
        true_params=[0.99, 0.1],
        noise_std=0.01,
        correlations=test_correlations
    )

    # 2. Trajektória: Alacsony fertőzőképesség, gyors gyógyulás (Lapos, széles görbe)
    data_test_2 = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0],
        true_params=[0.01, 0.1],
        noise_std=0.01,
        correlations=test_correlations
    )

    preprocessor = DataPreprocessor()
    cached_train_data = torch.load(
    os.path.join(
        OUTPUT_DIR,
        f"{NUM_STATES}_states_{SEQ_LEN}len_target_{TARGET_IDX}.pt"
    ),
    weights_only=False
    )
    preprocessor.x_scaler.fit(np.concatenate(cached_train_data["x_train"]))
    x_scaler = preprocessor.x_scaler
        
    x_test_scaled_1 = x_scaler.transform(data_test_1["x_raw"])
    x_test_scaled_2 = x_scaler.transform(data_test_2["x_raw"])
    
    true_target_1 = data_test_1["true_latent_states"][:, TARGET_IDX]
    true_target_2 = data_test_2["true_latent_states"][:, TARGET_IDX]
    
    full_t_raw = torch.arange(SEQ_LEN, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(DEVICE)
    
    
    # ==========================================
    # 4. EVALUATION & COMMON PCA
    # ==========================================
    print("\n--- 4. EVALUATION & COMMON PCA ---")

    # A korábbi 40/80/120/160 napos megnevezések nem egyeztek a t_end értékekkel.
    phases = [
        ("1. megfigyelési ablak", 5),
        ("2. megfigyelési ablak", 10),
        ("3. megfigyelési ablak", 25),
        ("4. megfigyelési ablak", 50),
    ]

    models = EINNBuilder.build_einn(
        model_cfg,
        train_cfg,
        {},
        "SIR",
        seed=42
    )

    models.time_module.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, "einn_time_module.pth"),
            map_location=DEVICE
        )
    )
    models.feature_module.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, "einn_feature_module.pth"),
            map_location=DEVICE
        )
    )
    models.output_module.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, "einn_output_module.pth"),
            map_location=DEVICE
        )
    )

    models.ode_model = SIRModel(target_idx=TARGET_IDX)

    evaluator = EINNEvaluator(models=models, device=DEVICE)

    # ------------------------------------------
    # 4.1. Run all inference experiments
    # ------------------------------------------
    phase_results = []

    for phase_name, t_end in phases:

        print(f"Evaluating: {phase_name}")

        results_1 = evaluator.evaluate(
            x_test=x_test_scaled_1,
            future_t=full_t_raw,
            scaling_stats=scaling_stats,
            target_idx=TARGET_IDX,
            true_target_trajectory=true_target_1,
            t_end=t_end
        )

        results_2 = evaluator.evaluate(
            x_test=x_test_scaled_2,
            future_t=full_t_raw,
            scaling_stats=scaling_stats,
            target_idx=TARGET_IDX,
            true_target_trajectory=true_target_2,
            t_end=t_end
        )

        phase_results.append({
            "phase_name": phase_name,
            "t_end": t_end,
            "results_1": results_1,
            "results_2": results_2
        })

    
    # ------------------------------------------
    # 4.2. Collect latent representations
    # ------------------------------------------
    latent_arrays_a = [
        phase["results_1"]["latent_feature"]
        for phase in phase_results
    ]

    latent_arrays_b = [
        phase["results_2"]["latent_feature"]
        for phase in phase_results
    ]

    all_embeddings_a = np.concatenate(latent_arrays_a, axis=0)
    all_embeddings_b = np.concatenate(latent_arrays_b, axis=0)

    all_embeddings = np.concatenate(
        [all_embeddings_a, all_embeddings_b],
        axis=0
    )

    print("\nLatent embedding matrices:")
    print(f"A shape: {all_embeddings_a.shape}")
    print(f"B shape: {all_embeddings_b.shape}")
    print(f"Combined shape: {all_embeddings.shape}")

    # ------------------------------------------
    # 4.3. Fit the three PCA transformations
    # ------------------------------------------

    # PCA fitted jointly on A and B
    pca_common = PCA(n_components=2)
    pca_common.fit(all_embeddings)

    # PCA fitted only on trajectory A
    pca_a = PCA(n_components=2)
    pca_a.fit(all_embeddings_a)

    # PCA fitted only on trajectory B
    pca_b = PCA(n_components=2)
    pca_b.fit(all_embeddings_b)

    # Explained variance of the common PCA
    pca_variance = PCA()
    pca_variance.fit(all_embeddings)

    cum_variance = np.cumsum(
        pca_variance.explained_variance_ratio_
    )

    n_comp_95 = int(np.argmax(cum_variance >= 0.95) + 1)

    print(
        f"Dimensions required for 95% variance "
        f"(common PCA): {n_comp_95}"
    )

    # ------------------------------------------
    # 4.4. Apply all PCA transformations
    # ------------------------------------------

    for phase in phase_results:

        emb_a = phase["results_1"]["latent_feature"]
        emb_b = phase["results_2"]["latent_feature"]

        # 1. Common PCA
        phase["pca_common_1"] = pca_common.transform(emb_a)
        phase["pca_common_2"] = pca_common.transform(emb_b)

        # 2. PCA fitted on A, applied to A and Bs
        phase["pca_a_1"] = pca_a.transform(emb_a)
        phase["pca_a_2"] = pca_a.transform(emb_b)

        # 3. PCA fitted on B, applied to A and B
        phase["pca_b_1"] = pca_b.transform(emb_a)
        phase["pca_b_2"] = pca_b.transform(emb_b)

    print("\nPCA transformations completed successfully.")


    # ==========================================
    # 5. PLOTTING – csak megjelenítés, a fenti tanítás és inference változatlan
    # ==========================================
    print("\n--- 5. PLOTTING ---")
    png_path, pdf_path = plot_dual_inference_results(
        phase_results=phase_results,
        true_target_1=true_target_1,
        true_target_2=true_target_2,
        output_dir=OUTPUT_DIR,
        embedding_dim=model_cfg.d_e,
    )
    print(f"\nÁbra mentve (PNG): {png_path}\nÁbra mentve (PDF): {pdf_path}")


if __name__ == "__main__":
    run_sir_dual_inference_experiment()