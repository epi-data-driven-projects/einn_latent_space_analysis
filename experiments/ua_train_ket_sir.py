import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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
NUM_STATES = 20
DEVICE = 'cuda:1'
OUTPUT_DIR = "./experiment_results/sir_dual_inference"


def run_sir_dual_inference_experiment():
    """
    Executes the full pipeline for the SIR Epidemic Model, then evaluates
    two drastically different SIR trajectories side-by-side on the 4x6 plot.
    """
    ExperimentUtils.seed_everything(seed=42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_name = f"Compartment_{TARGET_IDX}"

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
        "beta": (0.3, 0.7),
        "gamma": (0.05, 0.2)
    }

    dataset, scaling_stats = loader.prepare_training_data(
        num_states=NUM_STATES,
        seq_len=100,
        d_p=2,
        param_ranges=SIR_PARAM_RANGES,
        initial_states=[0.99, 0.01, 0.0],
        noise_std=0.01,
        window_step=40  # Gyorsabb lefutás kísérletezéshez
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
    generator_test = SyntheticDataGenerator(true_ode_model=true_model, seq_len=100)

    # 1. Trajektória: Magas fertőzőképesség, lassú gyógyulás (Nagy haranggörbe)
    data_test_1 = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0],
        true_params=[0.6, 0.08],
        noise_std=0.01,
        correlations=test_correlations
    )

    # 2. Trajektória: Alacsony fertőzőképesség, gyors gyógyulás (Lapos, széles görbe)
    data_test_2 = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0],
        true_params=[0.35, 0.18],
        noise_std=0.01,
        correlations=test_correlations
    )

    preprocessor = DataPreprocessor()
    cached_train_data = torch.load(os.path.join(OUTPUT_DIR, f"{NUM_STATES}_states_target_{TARGET_IDX}.pt"),
                                   weights_only=False)
    preprocessor.x_scaler.fit(np.concatenate(cached_train_data["x_train"]))
    x_scaler = preprocessor.x_scaler

    x_test_scaled_1 = x_scaler.transform(data_test_1["x_raw"])
    x_test_scaled_2 = x_scaler.transform(data_test_2["x_raw"])

    true_target_1 = data_test_1["true_latent_states"][:, TARGET_IDX]
    true_target_2 = data_test_2["true_latent_states"][:, TARGET_IDX]

    full_t_raw = torch.arange(100, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(DEVICE)

    # ==========================================
    # 4. EVALUATION & 4x6 PLOTTING (DUAL INFERENCE)
    # ==========================================
    print("\n--- 4. EVALUATION & PLOTTING ---")

    # A fázisokat (ablakokat) fixen 40 napos lépésekben határozzuk meg az összehasonlíthatóságért
    phases = [
        ("1. Kezdeti szakasz (40. nap)", 5),
        ("2. Felszálló ág (80. nap)", 10),
        ("3. Csúcs környéke (120. nap)", 25),
        ("4. Lecsengés (160. nap)", 50)
    ]

    fig = plt.figure(figsize=(32, 20))
    fig.suptitle(f"EINN Dual Inferencia ({target_name}): Két eltérő SIR paraméterezés összehasonlítása", fontsize=24,
                 y=0.98)

    models = EINNBuilder.build_einn(model_cfg, train_cfg, {}, "SIR", seed=42)
    models.time_module.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "einn_time_module.pth")))
    models.feature_module.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "einn_feature_module.pth")))
    models.output_module.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "einn_output_module.pth")))
    models.ode_model = SIRModel(target_idx=TARGET_IDX)

    evaluator = EINNEvaluator(models=models, device=DEVICE)

    for idx, (phase_name, t_end) in enumerate(phases):

        # Kiértékeljük az első görbét
        results_1 = evaluator.evaluate(
            x_test=x_test_scaled_1, future_t=full_t_raw, scaling_stats=scaling_stats,
            target_idx=TARGET_IDX, true_target_trajectory=true_target_1, t_end=t_end
        )

        # Kiértékeljük a második görbét
        results_2 = evaluator.evaluate(
            x_test=x_test_scaled_2, future_t=full_t_raw, scaling_stats=scaling_stats,
            target_idx=TARGET_IDX, true_target_trajectory=true_target_2, t_end=t_end
        )

        pred_np_1 = results_1["predicted_trajectory"]
        pred_np_2 = results_2["predicted_trajectory"]

        pca_coords_1 = results_1["pca_coords_3d"]
        pca_coords_2 = results_2["pca_coords_3d"]

        cos_sim_1 = results_1["cosine_similarity"]
        cos_sim_2 = results_2["cosine_similarity"]

        # Subplotok inicializálása (Soronként 6 ábra)
        ax_pred = fig.add_subplot(4, 6, idx * 6 + 1)
        ax_tsne = fig.add_subplot(4, 6,
                                  idx * 6 + 2)  # Ez megmarad a struktúra miatt, de üres lesz vagy csak 1-et ábrázol
        ax_pca_var = fig.add_subplot(4, 6, idx * 6 + 3)
        ax_pca_2d = fig.add_subplot(4, 6, idx * 6 + 4)
        ax_pca_3d = fig.add_subplot(4, 6, idx * 6 + 5, projection='3d')
        ax_cos = fig.add_subplot(4, 6, idx * 6 + 6)

        # 2a. Prediction Curve (Mindkét modell)
        ax_pred.plot(true_target_1, label='Modell 1 (Gyors) Valós', color='black', linewidth=2)
        ax_pred.plot(pred_np_1, label=f'Modell 1 Pred (MSE: {results_1["mse_future"]:.4f})', color='red',
                     linestyle='--', linewidth=2)

        ax_pred.plot(true_target_2, label='Modell 2 (Lassú) Valós', color='gray', linewidth=2)
        ax_pred.plot(pred_np_2, label=f'Modell 2 Pred (MSE: {results_2["mse_future"]:.4f})', color='orange',
                     linestyle='--', linewidth=2)

        ax_pred.axvline(x=t_end, color='blue', linestyle=':', linewidth=1.5, label='Ablak Vége')
        ax_pred.axvspan(0, t_end, color='blue', alpha=0.05)
        ax_pred.set_title(f"{phase_name}")
        ax_pred.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_pred.legend(loc='upper right', fontsize=8)

        # 2b. PCA 2D (Mindkét modell latens pályája)
        # Modell 1 (Piros)
        ax_pca_2d.plot(pca_coords_1[:, 0], pca_coords_1[:, 1], color='red', alpha=0.6, linewidth=2,
                       label="Modell 1 Pálya")
        ax_pca_2d.scatter(pca_coords_1[t_end - 1, 0], pca_coords_1[t_end - 1, 1], color='red', marker='o', s=100,
                          zorder=5)

        # Modell 2 (Kék)
        ax_pca_2d.plot(pca_coords_2[:, 0], pca_coords_2[:, 1], color='blue', alpha=0.6, linewidth=2,
                       label="Modell 2 Pálya")
        ax_pca_2d.scatter(pca_coords_2[t_end - 1, 0], pca_coords_2[t_end - 1, 1], color='blue', marker='o', s=100,
                          zorder=5)

        ax_pca_2d.set_title("PCA Látens Tér (2D) - Összehasonlítás")
        ax_pca_2d.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_pca_2d.legend()

        # 2c. PCA 3D (Mindkét modell)
        ax_pca_3d.plot(pca_coords_1[:, 0], pca_coords_1[:, 1], pca_coords_1[:, 2], color='red', alpha=0.6, linewidth=2)
        ax_pca_3d.scatter(pca_coords_1[t_end - 1, 0], pca_coords_1[t_end - 1, 1], pca_coords_1[t_end - 1, 2],
                          color='red', marker='o', s=100)

        ax_pca_3d.plot(pca_coords_2[:, 0], pca_coords_2[:, 1], pca_coords_2[:, 2], color='blue', alpha=0.6, linewidth=2)
        ax_pca_3d.scatter(pca_coords_2[t_end - 1, 0], pca_coords_2[t_end - 1, 1], pca_coords_2[t_end - 1, 2],
                          color='blue', marker='o', s=100)

        ax_pca_3d.set_title("PCA Látens Tér (3D) - Összehasonlítás")

        # 2d. Cosine Similarity (Mindkét modell)
        ax_cos.plot(cos_sim_1, color='red', linewidth=2, label="Modell 1")
        ax_cos.plot(cos_sim_2, color='blue', linewidth=2, label="Modell 2")
        ax_cos.set_title("Time vs Feature Cosine Similarity")
        ax_cos.set_xlabel("t")
        ax_cos.set_ylim([-1.1, 1.1])
        ax_cos.grid(True, linestyle='--', alpha=0.6)
        if idx == 0: ax_cos.legend()

        # A maradék két ábrát (t-SNE, PCA Variancia) csak az első modellre rajzoljuk ki informatív jelleggel
        emb_tsne = results_1["tsne_coords"]
        cum_variance = results_1["pca_variance"]

        ax_tsne.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=np.arange(100), cmap='viridis', s=20)
        ax_tsne.plot(emb_tsne[:, 0], emb_tsne[:, 1], color='gray', alpha=0.3, linewidth=1)
        ax_tsne.scatter(emb_tsne[t_end - 1, 0], emb_tsne[t_end - 1, 1], color='blue', marker='o', s=80)
        ax_tsne.set_title("t-SNE (Csak Modell 1)")

        ax_pca_var.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', color='teal')
        ax_pca_var.axhline(y=0.95, color='r', linestyle='--')
        ax_pca_var.set_title("PCA Variancia (Csak Modell 1)")

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    filepath = os.path.join(OUTPUT_DIR, "sir_dual_inference_analysis.png")
    fig.savefig(filepath, dpi=300)

    print(f"\nExperiment finished! Dual analysis plot saved at:\n{filepath}")


if __name__ == "__main__":
    run_sir_dual_inference_experiment()
