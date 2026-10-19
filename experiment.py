import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_preprocessor import DataPreprocessor
from einn.data.einn_dataset import EINNDataset
from einn.data.ode_precalibrator import ODEPreCalibrator
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.model.einn_builder import EINNBuilder
from einn.ode.seirm_model import SEIRMModel
from einn.training.einn_trainer import EINNTrainer

def run_latent_space_experiment():
    output_dir = "./experiment_results/latent_dynamics"
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda:1'

    # ==========================================
    # 1. ADATGENERÁLÁS ÉS ELŐKÉSZÍTÉS
    # ==========================================
    print("Generáljuk a teljes 200 napos SEIRM valóságot...")
    true_model = SEIRMModel(target_idx=4)
    generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=200)
    data_dict = generator.generate(
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
        true_params=[0.4, 0.2, 0.1, 0.05],
        noise_std=0.02
    )
    
    full_x_raw = data_dict["x_raw"]
    full_y_raw = data_dict["y_raw"]
    true_states = data_dict["true_latent_states"]
    true_i = true_states[:, 2]  # I (Infected) rekesz

    # ==========================================
    # 2. A MODELL EGYSZERI BETANÍTÁSA A TELJES ADATON
    # ==========================================
    print("\n--- Modell betanítása a teljes 200 napos adatsoron ---")
    assumed_model = SEIRMModel(target_idx=4)
    calibrator = ODEPreCalibrator(assumed_ode_model=assumed_model, d_p=4, device=device)
    
    # Kalibrálás a teljes 200 napra
    aux_raw = calibrator.calibrate(y_raw=full_y_raw, initial_states=[0.99, 0.01, 0.0, 0.0, 0.0], epochs=100, lr=0.01)
    
    preprocessor = DataPreprocessor()
    batch_dict = preprocessor.fit_transform(x_raw=full_x_raw, y_raw=full_y_raw, aux_raw=aux_raw, target_idx=4)

    train_cfg = EINNTrainConfig(
        device=device,
        learning_rate=0.002,
        epochs_per_phase=300, # A 300 epoch itt már elég, mert 14 napos csúszóablakokat használunk!
        early_stopping_patience=20,
        loss_weights={
            # A Célváltozó (M) és a Mankó (S, E, I, R) tartja a helyén a görbéket!
            'data_T': 5.0, 'data_F': 5.0,
            'aux': 4.0,  # VISSZAÁLLÍTVA! Ez kényszeríti ki a haranggörbéket!

            # A fizika finomhangolja, hogy a deriváltak matematikailag is stimmeljenek
            'ode_T': 6.0, 'ode_F': 4.0,
            'ode_future_T': 1.0, 'ode_future_F': 1.0,

            # Szigorú monotonitás (S csak lefelé, R és M csak felfelé mehet)
            'mono': 2.0,
            'param': 0.5,

            'kd_target': 1.0, 'kd_emb': 1.0,
            'ic': 2.0, 'ic_F': 0.0
        }
    )
    
    # Statisztikák GPU-ra helyezése
    train_cfg.aux_mean = preprocessor.aux_mean.to(device)
    train_cfg.aux_std = preprocessor.aux_std.to(device)
    train_cfg.t_max = preprocessor.t_max.to(device)
    
    # IC skálázás
    ic_physical = torch.tensor([0.99, 0.01, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    ic_normalized = (ic_physical - train_cfg.aux_mean) / (train_cfg.aux_std + 1e-8)
    train_cfg.initial_states = ic_normalized.cpu().tolist()

    # Csúszóablakos dataset a teljes adatsoron!
    dataset = EINNDataset(
        x=batch_dict["x"], y=batch_dict["y"], t=batch_dict["t"], 
        aux_targets=batch_dict["aux_targets"], window_size=200, device=device
    )

    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=5, d_p=4, feature_n_layers=2)
    models = EINNBuilder.build_einn(
        model_config=model_cfg, train_config=train_cfg,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
        model_type="SEIRM", seed=42
    )
    models.ode_model.target_idx = 4

    trainer = EINNTrainer(models=models, config=train_cfg)
    trainer.train(dataset=dataset, epochs=200, batch_size=1)

    # ==========================================
    # 3. JÁRVÁNYFÁZISOK DEFINIÁLÁSA (INFERENCIA)
    # ==========================================
    peak_day = int(np.argmax(true_i))
    peak_val = true_i[peak_day]
    
    rise_day = np.where((true_i >= peak_val * 0.2) & (np.arange(200) < peak_day))[0][0]
    decay_day = np.where((true_i <= peak_val * 0.5) & (np.arange(200) > peak_day))[0][0]
    
    phases = [
        ("1. Kezdeti szakasz (20 nap)", 20),
        (f"2. Felszálló ág ({rise_day}. nap)", rise_day),
        (f"3. Járványcsúcs ({peak_day}. nap)", peak_day),
        (f"4. Lecsengés fele ({decay_day}. nap)", decay_day),
        ("5. Teljes lecsengés (195. nap)", 195)
    ]

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 25))
    fig.suptitle("EINN Inferencia: Különböző Hosszúságú Bemenetek Hatása", fontsize=20, y=0.98)

    models.time_module.eval()
    models.feature_module.eval()
    models.output_module.eval()

    for idx, (phase_name, t_end) in enumerate(phases):
        print(f"\nInferencia futtatása: {phase_name}...")
        
        with torch.no_grad():
            # A hálózat csak t_end napnyi múltat kap meg bemenetként! (A TE ÖTLETED ALAPJÁN)
            hist_x_norm = batch_dict["x"][:t_end].unsqueeze(0).to(device)
            
            # De a teljes 200 napra kérjük a kimenetet
            full_t_raw = torch.arange(200, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(device)
            future_t_scaled = full_t_raw / (train_cfg.t_max + 1e-8)
            
            # Kinyerjük a Látens Tér Trajektóriáját (e_t_F)
            e_t_f = models.feature_module(x=hist_x_norm, t=future_t_scaled, mask=None)
            
            # Dekódolás fizikai rekeszekké
            states_norm = models.output_module(e=e_t_f)
            states_pred = (states_norm * train_cfg.aux_std) + train_cfg.aux_mean

        pred_np = states_pred.squeeze(0).cpu().numpy()
        emb_np = e_t_f.squeeze(0).cpu().numpy()

        # Hibaszámítás a jövőbeli (nem látott) szakaszon
        if t_end < 200:
            mse_future = mean_squared_error(true_states[t_end:, 2], pred_np[t_end:, 2])
        else:
            mse_future = mean_squared_error(true_states[:, 2], pred_np[:, 2])

        # --- Bal oszlop: Predikció ---
        ax_pred = axes[idx, 0]
        ax_pred.plot(true_i, label='Valós I (Infected)', color='black', linewidth=2)
        ax_pred.plot(pred_np[:, 2], label='Prediktált I', color='red', linestyle='--', linewidth=2)
        
        # Bemeneti ablak határvonalának jelzése
        ax_pred.axvline(x=t_end, color='blue', linestyle=':', linewidth=1.5, label='Múlt (Bemenet) Vége')
        ax_pred.fill_betweenx([0, max(true_i)*1.2], 0, t_end, color='blue', alpha=0.05)
        
        ax_pred.set_title(f"{phase_name}\nJövőbeli MSE: {mse_future:.2e}")
        ax_pred.set_ylabel("Infected Arány")
        ax_pred.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_pred.legend()

        # --- Jobb oszlop: t-SNE Látens Tér ---
        ax_tsne = axes[idx, 1]
        
        tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
        emb_2d = tsne.fit_transform(emb_np)
        
        # Pontok színezése az idő múlása szerint
        scatter = ax_tsne.scatter(
            emb_2d[:, 0], emb_2d[:, 1], 
            c=np.arange(200), cmap='viridis', 
            s=20, edgecolor='k', alpha=0.7
        )
        
        ax_tsne.plot(emb_2d[:, 0], emb_2d[:, 1], color='gray', alpha=0.3, linewidth=1)
        ax_tsne.scatter(emb_2d[0, 0], emb_2d[0, 1], color='red', marker='s', s=80, label='Start (0. nap)', zorder=5)
        ax_tsne.scatter(emb_2d[-1, 0], emb_2d[-1, 1], color='black', marker='X', s=100, label='End (200. nap)', zorder=5)
        ax_tsne.scatter(emb_2d[t_end-1, 0], emb_2d[t_end-1, 1], color='blue', marker='o', s=80, label=f'Bemenet Vége ({t_end}. nap)', zorder=5)
        
        ax_tsne.set_title(f"t-SNE Látens Tér ({model_cfg.d_e}D -> 2D)")
        ax_tsne.set_xlabel("t-SNE 1")
        ax_tsne.set_ylabel("t-SNE 2")
        ax_tsne.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_tsne.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filepath = os.path.join(output_dir, "latent_dynamics_inference.png")
    plt.savefig(filepath, dpi=300)
    print(f"\nA teljes kísérlet lefutott. Az összevont ábra elmentve ide: {filepath}")

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    run_latent_space_experiment()
