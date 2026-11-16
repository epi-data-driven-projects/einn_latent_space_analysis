import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.einn_input_loader import EINNInputLoader
from einn.data.synthetic_data_generator import SyntheticDataGenerator
# JAVÍTVA: Importáljuk a Röst modellt a megfelelő fájlból
from einn.ode.rost_model import RostHungaryModel
from einn_runner import EINNRunner
from einn.inference.einn_evaluator import EINNEvaluator
from einn.utils.experiment_utils import ExperimentUtils
from einn.model.einn_builder import EINNBuilder

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
# JAVÍTVA: A Röst modell 15-ös indexe a Kumulatív Fertőzöttek (Cumulative Infected)
TARGET_IDX = 15
NUM_STATES = 20
DEVICE = 'cuda:1'
OUTPUT_DIR = "./experiment_results/generalization"


def run_generalization_experiment():
    """
    Executes the full pipeline for the Röst Epidemic Model.
    """
    # 1. ENFORCE DETERMINISM
    ExperimentUtils.seed_everything(seed=42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_name = "Cumulative Infected" if TARGET_IDX == 15 else f"Compartment_{TARGET_IDX}"

    # ==========================================
    # 2. DATA PREPARATION & TRAINING
    # ==========================================
    print("--- 1. DATA PREPARATION ---")
    loader = EINNInputLoader(
        true_model=RostHungaryModel(target_idx=TARGET_IDX),  # JAVÍTVA
        calibrator_model=RostHungaryModel(target_idx=TARGET_IDX),  # JAVÍTVA
        target_idx=TARGET_IDX,
        device=DEVICE,
        cache_dir=OUTPUT_DIR
    )

    ROST_PARAM_RANGES = {
        "beta": (0.2, 0.6),
        "inf_a": (0.2, 0.6),
        "susc": (0.7, 1.0),
        "alpha_l": (0.1, 0.3),
        "alpha_p": (0.1, 0.4),
        "p": (0.2, 0.8),
        "gamma_a": (0.05, 0.2),
        "gamma_s": (0.05, 0.2),
        "h": (0.02, 0.15),
        "xi": (0.02, 0.2),
        "gamma_h": (0.05, 0.2),
        "gamma_c": (0.02, 0.1),
        "mu": (0.01, 0.1),
        "gamma_cr": (0.05, 0.2)
    }

    dataset, scaling_stats = loader.prepare_training_data(
        num_states=NUM_STATES,
        seq_len=200,
        d_p=14,  # JAVÍTVA: 4 helyett 14 fizikai paraméter
        param_ranges=ROST_PARAM_RANGES,
        noise_std=0.0005
    )

    print(f"Dinamikus t_min: {scaling_stats.t_min.item()}")
    print(f"Dinamikus t_max: {scaling_stats.t_max.item()}")

    print("\n--- 2. MODEL TRAINING ---")
    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=18, d_p=14, feature_n_layers=2)

    train_cfg = EINNTrainConfig(
        device=DEVICE, learning_rate=0.001, early_stopping_patience=5,
        loss_weights={
            'data_T': 10.0, 'data_F': 10.0,
            'aux': 4.0,
            'ode_T': 8.0, 'ode_F': 8.0,
            'ode_future_T': 0.1, 'ode_future_F': 0.1,
            'mono': 3.0,
            'param': 0.1,
            'kd_target': 10.0, 'kd_emb': 10.0,
            'ic': 20.0, 'ic_F': 20.0
        },
        reps={'phase_1': 100, 'phase_2': 100, 'phase_3': 100, 'phase_4': 100}
    )

    rost_calibration = {
        "beta": 0.4, "inf_a": 0.5, "susc": 0.9, "alpha_l": 0.2,
        "alpha_p": 0.3, "p": 0.4, "gamma_a": 0.1, "gamma_s": 0.1,
        "h": 0.05, "xi": 0.1, "gamma_h": 0.1, "gamma_c": 0.05,
        "mu": 0.02, "gamma_cr": 0.1
    }

    runner = EINNRunner(output_dir=OUTPUT_DIR, model_cfg=model_cfg, train_cfg=train_cfg, seed=42)
    runner.train_and_save(
        dataset=dataset,
        scaling_stats=scaling_stats,
        model_type="ROST",  # FONTOS: Az EINNBuilder-ben ennek "ROST"-ként kell szerepelnie!
        target_idx=TARGET_IDX,
        param_calibration=rost_calibration,  # JAVÍTVA
        epochs=3,
        batch_size=32
    )

    # ==========================================
    # 3. GENERATE UNSEEN TEST DATA (STATE 51)
    # ==========================================
    print("\n--- 3. GENERATING UNSEEN TEST STATE ---")
    test_correlations = {
        'mobility': -0.75, 'temperature': -0.1, 'testing': 0.85, 'stringency': 0.95
    }

    true_model = RostHungaryModel(target_idx=TARGET_IDX)
    generator_test = SyntheticDataGenerator(true_ode_model=true_model, seq_len=200)

    # JAVÍTVA: 18 elemű kezdeti állapot (S=0.99, L1=0.01, többi nulla) és 14 elemű true_params vektor
    data_test = generator_test.generate(
        initial_states=[0.99, 0.01] + [0.0] * 16,
        true_params=[0.45, 0.4, 0.95, 0.25, 0.25, 0.5, 0.15, 0.15, 0.1, 0.08, 0.12, 0.05, 0.03, 0.1],
        noise_std=0.0005,
        correlations=test_correlations
    )

    from einn.data.data_preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    cached_train_data = torch.load(os.path.join(OUTPUT_DIR, f"{NUM_STATES}_states_target_{TARGET_IDX}.pt"),
                                   weights_only=False)
    preprocessor.x_scaler.fit(np.concatenate(cached_train_data["x_train"]))
    x_scaler = preprocessor.x_scaler

    x_test_scaled = x_scaler.transform(data_test["x_raw"])
    true_target = data_test["true_latent_states"][:, TARGET_IDX]

    # Calculate full scaled time vector
    full_t_raw = torch.arange(200, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(DEVICE)

    # ==========================================
    # 4. EVALUATION & 4x6 PLOTTING
    # ==========================================
    print("\n--- 4. EVALUATION & PLOTTING ---")

    # JAVÍTVA: Mivel a Kumulatív Fertőzöttek (idx=15) folytonosan nő, a "csúcs" számításához
    # deriválnunk kell (vagyis az napi incidenciát, új eseteket kell vizsgálnunk).
    true_incidence = np.diff(true_target, prepend=0)

    peak_day = int(np.argmax(true_incidence))
    peak_val = true_incidence[peak_day]

    # Keresünk egy 20%-os felszálló és egy 50%-os lecsengő napot az INICIDENCIÁBÓL
    try:
        rise_day = np.where((true_incidence >= peak_val * 0.2) & (np.arange(200) < peak_day))[0][0]
    except IndexError:
        rise_day = max(1, peak_day - 15)

    try:
        decay_day = np.where((true_incidence <= peak_val * 0.5) & (np.arange(200) > peak_day))[0][0]
    except IndexError:
        decay_day = min(199, peak_day + 20)

    phases = [
        (f"1. Felszallo ag ({rise_day}. nap)", rise_day),
        (f"2. Jarvanycsucs ({peak_day}. nap)", peak_day),
        (f"3. Lecsenges ({decay_day}. nap)", decay_day),
        ("4. Jarvany vege (150. nap)", 150)
    ]

    fig = plt.figure(figsize=(32, 20))
    fig.suptitle(f"EINN Generalizacio: Inferencia ({target_name}) es Latens Ter Evolucio", fontsize=24, y=0.98)

    # Initialize Evaluator with the recently trained models
    # FONTOS JAVÍTÁS: Az EINNBuilder hívásnál "ROST" modelltípust kell átadni!
    models = EINNBuilder.build_einn(model_cfg, train_cfg, {}, "ROST", seed=42)
    models.time_module.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "einn_time_module.pth")))
    models.feature_module.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "einn_feature_module.pth")))
    models.output_module.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "einn_output_module.pth")))

    models.ode_model = RostHungaryModel(target_idx=TARGET_IDX)

    evaluator = EINNEvaluator(models=models, device=DEVICE)

    for idx, (phase_name, t_end) in enumerate(phases):
        results = evaluator.evaluate(
            x_test=x_test_scaled,
            future_t=full_t_raw,
            scaling_stats=scaling_stats,
            target_idx=TARGET_IDX,
            true_target_trajectory=true_target,
            t_end=t_end
        )

        pred_np = results["predicted_trajectory"]
        emb_tsne = results["tsne_coords"]
        cum_variance = results["pca_variance"]
        n_comp = results["pca_dims_95"] if "pca_dims_95" in results else results.get("pca_dims_99", 3)
        pca_coords = results["pca_coords_3d"]
        disparity = results["procrustes_disparity"]
        cos_sim = results["cosine_similarity"]

        # 2. Add 6 Subplots per row
        ax_pred = fig.add_subplot(4, 6, idx * 6 + 1)
        ax_tsne = fig.add_subplot(4, 6, idx * 6 + 2)
        ax_pca_var = fig.add_subplot(4, 6, idx * 6 + 3)
        ax_pca_2d = fig.add_subplot(4, 6, idx * 6 + 4)
        ax_pca_3d = fig.add_subplot(4, 6, idx * 6 + 5, projection='3d')
        ax_cos = fig.add_subplot(4, 6, idx * 6 + 6)

        # 2a. Prediction Curve
        ax_pred.plot(true_target, label=f'Valos {target_name}', color='black', linewidth=2)
        ax_pred.plot(pred_np, label=f'Prediktalt (MSE: {results["mse_future"]:.4f})', color='red', linestyle='--',
                     linewidth=2)
        ax_pred.axvline(x=t_end, color='blue', linestyle=':', linewidth=1.5, label='Ablak Vege')
        ax_pred.axvspan(0, t_end, color='blue', alpha=0.05)
        ax_pred.set_title(f"{phase_name}")
        ax_pred.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_pred.legend()

        # 2b. t-SNE
        ax_tsne.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=np.arange(200), cmap='viridis', s=20, edgecolor='k',
                        alpha=0.7)
        ax_tsne.plot(emb_tsne[:, 0], emb_tsne[:, 1], color='gray', alpha=0.3, linewidth=1)
        ax_tsne.scatter(emb_tsne[0, 0], emb_tsne[0, 1], color='red', marker='s', s=80, label='Start', zorder=5)
        ax_tsne.scatter(emb_tsne[-1, 0], emb_tsne[-1, 1], color='black', marker='X', s=100, label='End', zorder=5)
        ax_tsne.scatter(emb_tsne[t_end - 1, 0], emb_tsne[t_end - 1, 1], color='blue', marker='o', s=80, label='Jelen',
                        zorder=5)
        ax_tsne.set_title(f"t-SNE (20D -> 2D)")
        ax_tsne.grid(True, linestyle='--', alpha=0.5)

        # 2c. PCA Variance
        ax_pca_var.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='-', color='teal')
        ax_pca_var.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax_pca_var.axhline(y=0.99, color='g', linestyle='--', label='99%')
        ax_pca_var.set_title(f"PCA Variancia (Dimenziok: {n_comp})")
        ax_pca_var.set_xlabel("Fokomponensek")
        ax_pca_var.grid(True, linestyle='--', alpha=0.6)

        # 2d. PCA 2D
        ax_pca_2d.scatter(pca_coords[:, 0], pca_coords[:, 1], c=np.arange(200), cmap='viridis', s=20, edgecolor='k',
                          alpha=0.7)
        ax_pca_2d.plot(pca_coords[:, 0], pca_coords[:, 1], color='gray', alpha=0.3, linewidth=1)
        ax_pca_2d.scatter(pca_coords[0, 0], pca_coords[0, 1], color='red', marker='s', s=80, zorder=5)
        ax_pca_2d.scatter(pca_coords[-1, 0], pca_coords[-1, 1], color='black', marker='X', s=100, zorder=5)
        ax_pca_2d.scatter(pca_coords[t_end - 1, 0], pca_coords[t_end - 1, 1], color='blue', marker='o', s=80, zorder=5)
        ax_pca_2d.set_title("PCA Latens Ter (2D)")
        ax_pca_2d.grid(True, linestyle='--', alpha=0.5)

        # 2e. PCA 3D
        ax_pca_3d.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2], c=np.arange(200), cmap='viridis', s=20,
                          edgecolor='k', alpha=0.7)
        ax_pca_3d.plot(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2], color='gray', alpha=0.3, linewidth=1)
        ax_pca_3d.scatter(pca_coords[0, 0], pca_coords[0, 1], pca_coords[0, 2], color='red', marker='s', s=80, zorder=5)
        ax_pca_3d.scatter(pca_coords[-1, 0], pca_coords[-1, 1], pca_coords[-1, 2], color='black', marker='X', s=100,
                          zorder=5)
        ax_pca_3d.scatter(pca_coords[t_end - 1, 0], pca_coords[t_end - 1, 1], pca_coords[t_end - 1, 2], color='blue',
                          marker='o', s=80, zorder=5)
        ax_pca_3d.set_title("PCA Latens Ter (3D)")
        ax_pca_3d.xaxis.pane.fill = False
        ax_pca_3d.yaxis.pane.fill = False
        ax_pca_3d.zaxis.pane.fill = False

        # 2f. Cosine Similarity & Procrustes
        ax_cos.plot(cos_sim, color='purple', linewidth=2)
        ax_cos.set_title(f"Time vs Feature (Procrustes: {disparity:.4f})")
        ax_cos.set_xlabel("t")
        ax_cos.set_ylim([-1.1, 1.1])
        ax_cos.grid(True, linestyle='--', alpha=0.6)

    # Save the massive 4x6 figure
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    filepath = os.path.join(OUTPUT_DIR, f"rost_generalization_full_analysis_target_{TARGET_IDX}.png")
    fig.savefig(filepath, dpi=300)

    print(f"\nExperiment finished! Full analysis plot saved at:\n{filepath}")


if __name__ == "__main__":
    run_generalization_experiment()
