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
TARGET_IDX = 1  # SIR Infected
NUM_STATES = 20 # Kisebb szám a gyorsabb dupla tanításért
SEQ_LEN = 100   # Globális változó
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

DIR_FAST = "./experiment_results/sir_model_fast"
DIR_SLOW = "./experiment_results/sir_model_slow"
DIR_OUT = "./experiment_results/dual_model_comparison"


def train_model_regime(regime_name, output_dir, param_ranges):
    """Betanít egy modellt a megadott paramétertartományon."""
    print(f"\n{'='*50}\n TANÍTÁS: {regime_name} REZSIM\n{'='*50}")
    os.makedirs(output_dir, exist_ok=True)
    
    loader = EINNInputLoader(
        true_model=SIRModel(target_idx=TARGET_IDX),
        calibrator_model=SIRModel(target_idx=TARGET_IDX),
        target_idx=TARGET_IDX,
        device=DEVICE,
        cache_dir=output_dir
    )

    dataset, scaling_stats = loader.prepare_training_data(
        num_states=NUM_STATES, 
        seq_len=SEQ_LEN, 
        d_p=2, 
        param_ranges=param_ranges,
        initial_states=[0.99, 0.01, 0.0],
        noise_std=0.01,
        window_step=20
    )

    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=3, d_p=2, feature_n_layers=2)
    train_cfg = EINNTrainConfig(
        device=DEVICE, learning_rate=0.001, early_stopping_patience=5,
        loss_weights={
            'data_T': 10.0, 'data_F': 10.0, 'aux': 2.0, 'ode_T': 8.0, 'ode_F': 8.0,
            'ode_future_T': 5.1, 'ode_future_F': 5.1, 'mono': 1.0, 'param': 0.1,
            'kd_target': 5.0, 'kd_emb': 5.0, 'ic': 20.0, 'ic_F': 20.0
        },
        reps={'phase_1': 10, 'phase_2': 10, 'phase_3': 10, 'phase_4': 10}
    )

    runner = EINNRunner(output_dir=output_dir, model_cfg=model_cfg, train_cfg=train_cfg, seed=42)
    runner.train_and_save(
        dataset=dataset, scaling_stats=scaling_stats, model_type="SIR",
        target_idx=TARGET_IDX, param_calibration={"beta": 0.4, "gamma": 0.1},
        epochs=10, batch_size=64
    )
    return model_cfg, train_cfg, scaling_stats


def load_trained_model(output_dir, model_cfg, train_cfg):
    """Visszatölti a betanított modellt a lemezről."""
    models = EINNBuilder.build_einn(model_cfg, train_cfg, {}, "SIR", seed=42)
    models.time_module.load_state_dict(torch.load(os.path.join(output_dir, "einn_time_module.pth")))
    models.feature_module.load_state_dict(torch.load(os.path.join(output_dir, "einn_feature_module.pth")))
    models.output_module.load_state_dict(torch.load(os.path.join(output_dir, "einn_output_module.pth")))
    models.ode_model = SIRModel(target_idx=TARGET_IDX)
    return models


def load_model_scaler(output_dir):
    """Visszatölti az adott modell saját DataPreprocessor-át a cache-ből."""
    preprocessor = DataPreprocessor()
    cached_data = torch.load(os.path.join(output_dir, f"{NUM_STATES}_states_{SEQ_LEN}len_target_{TARGET_IDX}.pt"), weights_only=False)
    preprocessor.x_scaler.fit(np.concatenate(cached_data["x_train"]))
    return preprocessor.x_scaler


def run_dual_model_experiment():
    ExperimentUtils.seed_everything(seed=42)
    os.makedirs(DIR_OUT, exist_ok=True)

    # 1. Két modell betanítása eltérő adatokon
    fast_ranges = {"beta": (0.5, 0.8), "gamma": (0.1, 0.2)}
    cfg_fast, tcfg_fast, stats_fast = train_model_regime("FAST (Robbanásszerű)", DIR_FAST, fast_ranges)
    
    slow_ranges = {"beta": (0.15, 0.35), "gamma": (0.1, 0.2)}
    cfg_slow, tcfg_slow, stats_slow = train_model_regime("SLOW (Lapos)", DIR_SLOW, slow_ranges)

    # 2. Köztes, ismeretlen teszt adat generálása
    print("\n--- KÖZÖS TESZT ADAT GENERÁLÁSA ---")
    test_model = SIRModel(target_idx=TARGET_IDX)
    gen_test = SyntheticDataGenerator(true_ode_model=test_model, seq_len=SEQ_LEN)
    data_test = gen_test.generate(
        initial_states=[0.99, 0.01, 0.0],
        true_params=[0.42, 0.15],
        noise_std=0.01,
        correlations={'mobility': -0.7, 'temperature': -0.1, 'testing': 0.8, 'stringency': 0.9}
    )
    
    x_raw = data_test["x_raw"]
    true_target = data_test["true_latent_states"][:, TARGET_IDX]
    full_t_raw = torch.arange(SEQ_LEN, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(DEVICE)

    scaler_fast = load_model_scaler(DIR_FAST)
    scaler_slow = load_model_scaler(DIR_SLOW)
    
    x_test_fast = scaler_fast.transform(x_raw)
    x_test_slow = scaler_slow.transform(x_raw)

    models_fast = load_trained_model(DIR_FAST, cfg_fast, tcfg_fast)
    models_slow = load_trained_model(DIR_SLOW, cfg_slow, tcfg_slow)
    
    eval_fast = EINNEvaluator(models=models_fast, device=DEVICE)
    eval_slow = EINNEvaluator(models=models_slow, device=DEVICE)

    # 3. Kiértékelés és Ábrázolás
    print("\n--- INFERENCIA ÉS VIZUALIZÁCIÓ ---")
    p1 = int(SEQ_LEN * 0.2)
    p2 = int(SEQ_LEN * 0.4)
    p3 = int(SEQ_LEN * 0.6)
    p4 = int(SEQ_LEN * 0.8)
    
    phases = [
        (f"1. Kezdeti ({p1}. nap)", p1), (f"2. Felszálló ({p2}. nap)", p2),
        (f"3. Csúcs környéke ({p3}. nap)", p3), (f"4. Lecsengés ({p4}. nap)", p4)
    ]

    fig = plt.figure(figsize=(32, 20))

    for idx, (phase_name, t_end) in enumerate(phases):
        
        res_fast = eval_fast.evaluate(
            x_test=x_test_fast, future_t=full_t_raw, scaling_stats=stats_fast,
            target_idx=TARGET_IDX, true_target_trajectory=true_target, t_end=t_end
        )
        
        res_slow = eval_slow.evaluate(
            x_test=x_test_slow, future_t=full_t_raw, scaling_stats=stats_slow,
            target_idx=TARGET_IDX, true_target_trajectory=true_target, t_end=t_end
        )

        ax_pred    = fig.add_subplot(4, 6, idx * 6 + 1)
        ax_tsne    = fig.add_subplot(4, 6, idx * 6 + 2)
        ax_pca_var = fig.add_subplot(4, 6, idx * 6 + 3)
        ax_pca_2d  = fig.add_subplot(4, 6, idx * 6 + 4)
        ax_pca_3d  = fig.add_subplot(4, 6, idx * 6 + 5, projection='3d')
        ax_cos     = fig.add_subplot(4, 6, idx * 6 + 6)

        # 1. Predikciók
        ax_pred.plot(true_target, label='Valós Teszt Adat (Teljes)', color='lightgray', linewidth=2)
        ax_pred.plot(range(t_end), true_target[:t_end], color='black', linewidth=2.5)
        ax_pred.plot(res_fast["predicted_trajectory"], label=f'Modell A Pred', color='red', linestyle='--', linewidth=2)
        ax_pred.plot(res_slow["predicted_trajectory"], label=f'Modell B Pred', color='blue', linestyle='-.', linewidth=2)
        
        ax_pred.axvline(x=t_end, color='gray', linestyle=':', linewidth=1.5)
        ax_pred.axvspan(0, t_end, color='gray', alpha=0.1)
        ax_pred.set_title(f"{phase_name}")
        ax_pred.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_pred.legend(loc='upper right')

        # 2. PCA 2D (Teljes pálya ábrázolása, jelenlegi pont kiemelve fekete kerettel)
        pc_f, pc_s = res_fast["pca_coords_3d"], res_slow["pca_coords_3d"]
        ax_pca_2d.plot(pc_f[:, 0], pc_f[:, 1], color='red', alpha=0.4, linewidth=2, label="Modell A Pálya (Teljes)")
        ax_pca_2d.plot(pc_f[:t_end, 0], pc_f[:t_end, 1], color='red', alpha=0.9, linewidth=2)
        ax_pca_2d.scatter(pc_f[t_end - 1, 0], pc_f[t_end - 1, 1], color='red', marker='o', s=150, edgecolors='black', zorder=5)
        
        ax_pca_2d.plot(pc_s[:, 0], pc_s[:, 1], color='blue', alpha=0.4, linewidth=2, label="Modell B Pálya (Teljes)")
        ax_pca_2d.plot(pc_s[:t_end, 0], pc_s[:t_end, 1], color='blue', alpha=0.9, linewidth=2)
        ax_pca_2d.scatter(pc_s[t_end - 1, 0], pc_s[t_end - 1, 1], color='blue', marker='o', s=150, edgecolors='black', zorder=5)
        
        ax_pca_2d.set_title("PCA Látens Tér (2D) - 2 Modell Reakciója")
        ax_pca_2d.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_pca_2d.legend()

        # 3. PCA 3D (Teljes pálya ábrázolása)
        ax_pca_3d.plot(pc_f[:, 0], pc_f[:, 1], pc_f[:, 2], color='red', alpha=0.4, linewidth=2)
        ax_pca_3d.plot(pc_f[:t_end, 0], pc_f[:t_end, 1], pc_f[:t_end, 2], color='red', alpha=0.9, linewidth=2)
        ax_pca_3d.scatter(pc_f[t_end - 1, 0], pc_f[t_end - 1, 1], pc_f[t_end - 1, 2], color='red', marker='o', s=150, edgecolors='black', zorder=5)
        
        ax_pca_3d.plot(pc_s[:, 0], pc_s[:, 1], pc_s[:, 2], color='blue', alpha=0.4, linewidth=2)
        ax_pca_3d.plot(pc_s[:t_end, 0], pc_s[:t_end, 1], pc_s[:t_end, 2], color='blue', alpha=0.9, linewidth=2)
        ax_pca_3d.scatter(pc_s[t_end - 1, 0], pc_s[t_end - 1, 1], pc_s[t_end - 1, 2], color='blue', marker='o', s=150, edgecolors='black', zorder=5)
        ax_pca_3d.set_title("PCA Látens Tér (3D)")

        # 4. Cosine Similarity (Teljes pálya halványan, látott erősen)
        ax_cos.plot(range(SEQ_LEN), res_fast["cosine_similarity"], color='red', alpha=0.3, linewidth=2)
        ax_cos.plot(range(t_end), res_fast["cosine_similarity"][:t_end], color='red', linewidth=2.5, label="Modell A (Látott)")
        
        ax_cos.plot(range(SEQ_LEN), res_slow["cosine_similarity"], color='blue', alpha=0.3, linewidth=2)
        ax_cos.plot(range(t_end), res_slow["cosine_similarity"][:t_end], color='blue', linewidth=2.5, label="Modell B (Látott)")
        
        ax_cos.set_title(f"Time vs Feature Cosine Similarity\nProcrustes disparity (Fast): {res_fast['procrustes_disparity']:.4f}\nProcrustes disparity (Slow): {res_slow['procrustes_disparity']:.4f}")
        ax_cos.set_ylim([-1.1, 1.1])
        ax_cos.set_xlim([0, SEQ_LEN])
        ax_cos.grid(True, linestyle='--', alpha=0.6)
        if idx == 0: ax_cos.legend()

        # 5. t-SNE (Mindkét modell, teljes pálya)
        tsne_f = res_fast["tsne_coords"]
        tsne_s = res_slow["tsne_coords"]
        
        ax_tsne.plot(tsne_f[:, 0], tsne_f[:, 1], color='red', alpha=0.2, linewidth=1)
        ax_tsne.scatter(tsne_f[:, 0], tsne_f[:, 1], color='red', s=20, alpha=0.4, label="Modell A")
        ax_tsne.scatter(tsne_f[t_end - 1, 0], tsne_f[t_end - 1, 1], color='red', marker='o', s=120, edgecolors='black', zorder=5)
        
        ax_tsne.plot(tsne_s[:, 0], tsne_s[:, 1], color='blue', alpha=0.2, linewidth=1)
        ax_tsne.scatter(tsne_s[:, 0], tsne_s[:, 1], color='blue', s=20, alpha=0.4, label="Modell B")
        ax_tsne.scatter(tsne_s[t_end - 1, 0], tsne_s[t_end - 1, 1], color='blue', marker='o', s=120, edgecolors='black', zorder=5)
        
        ax_tsne.set_title("t-SNE (Mindkét Modell)")
        if idx == 0: ax_tsne.legend()
        
        # 6. PCA Variancia (Mindkét modell)
        ax_pca_var.plot(range(1, len(res_fast["pca_variance"]) + 1), res_fast["pca_variance"], marker='o', color='red', label="Modell A")
        ax_pca_var.plot(range(1, len(res_slow["pca_variance"]) + 1), res_slow["pca_variance"], marker='x', color='blue', label="Modell B")
        ax_pca_var.axhline(y=0.95, color='black', linestyle='--')
        ax_pca_var.set_title("PCA Variancia")
        if idx == 0: ax_pca_var.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    filepath = os.path.join(DIR_OUT, "two_models_one_data_analysis.png")
    fig.savefig(filepath, dpi=300)
    print(f"\nExperiment finished! Comparison plot saved at:\n{filepath}")

if __name__ == "__main__":
    run_dual_model_experiment()
    