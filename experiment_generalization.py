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
from einn.inference.inference_engine import InferenceEngine


TARGET_IDX = 2
NUM_STATES = 9  # Ezt később egyszerűen átírhatod 50-re!


class ExpandingWindowDataset(EINNDataset):
    def __init__(self, x_list, y_list, t_list, aux_list, min_days=10, max_days=200, step=10, device='cpu'):
        self.device = torch.device(device)
        self.samples = []

        for x, y, t, aux in zip(x_list, y_list, t_list, aux_list):
            for current_days in range(min_days, max_days + 1, step):
                # Bemenet paddingelése 200-ra (csak a DataLoader batch-elése miatt kell ide)
                x_padded = torch.zeros((max_days, x.shape[1]), device=device)
                x_padded[:current_days] = torch.tensor(x[:current_days], dtype=torch.float32, device=device)

                mask = torch.ones(max_days, dtype=torch.float32, device=device)
                mask[:current_days] = 0.0

                self.samples.append({
                    'x': x_padded,
                    'mask': mask,
                    'y': torch.tensor(y, dtype=torch.float32, device=device),
                    't': torch.tensor(t, dtype=torch.float32, device=device),
                    'aux_targets': torch.tensor(aux, dtype=torch.float32, device=device)
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def run_generalization_experiment():
    output_dir = "./experiment_results/generalization"
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda:1'
    
    data_cache_file = os.path.join(output_dir, "cached_seirm_data.pt")

    # ==========================================
    # 1. KÜLÖNBÖZŐ ÁLLAM GENERÁLÁSA VAGY BETÖLTÉSE
    # ==========================================
    true_model = SEIRMModel(target_idx=TARGET_IDX)
    calibrator_model = SEIRMModel(target_idx=TARGET_IDX)
    calibrator = ODEPreCalibrator(assumed_ode_model=calibrator_model, d_p=4, device=device)
    preprocessor = DataPreprocessor()

    x_train, y_train, aux_train = [], [], []
    generate_data = True

    if os.path.exists(data_cache_file):
        print(f"Cache fájl megtalálva: {data_cache_file}")
        cached_data = torch.load(data_cache_file, weights_only=False)
        
        if len(cached_data["x_train"]) <= NUM_STATES:
            print(f"Elegendő adat ({len(cached_data['x_train'])} állam) található a cache-ben. Betöltés...")
            x_train = cached_data["x_train"][:NUM_STATES]
            y_train = cached_data["y_train"][:NUM_STATES]
            aux_train = cached_data["aux_train"][:NUM_STATES]
            generate_data = False
        else:
            print(f"Nincs elég adat a cache-ben. Újragenerálás...")

    if generate_data:
        print(f"{NUM_STATES} különböző paraméterezésű SEIRM állam generálása (5D bemenettel) és kalibrálása...")
        for i in range(NUM_STATES):
            print(f"Jelenleg a {i+1}. generálásnál tartunk a {NUM_STATES}-ből...")
            beta = np.random.uniform(0.3, 0.6)
            alpha = np.random.uniform(0.1, 0.3)
            gamma = np.random.uniform(0.05, 0.15)
            mu = np.random.uniform(0.01, 0.08)

            # Dinamikus korrelációk minden államhoz (ettől lesz igazán generalizálható!)
            state_correlations = {
                'mobility': np.random.uniform(-0.9, -0.4),
                'temperature': np.random.uniform(-0.4, 0.1),
                'testing': np.random.uniform(0.5, 0.9),
                'stringency': np.random.uniform(0.7, 0.95)
            }

            generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=200)
            data = generator.generate(
                initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
                true_params=[beta, alpha, gamma, mu],
                noise_std=0.00,
                correlations=state_correlations
            )

            aux = calibrator.calibrate(y_raw=data["y_raw"], initial_states=[0.99, 0.01, 0.0, 0.0, 0.0], epochs=300, lr=0.001)

            x_train.append(data["x_raw"])
            y_train.append(data["y_raw"])

            aux[:, 4] = data["y_raw"].ravel()
            aux_train.append(aux)

        torch.save({
            "x_train": x_train,
            "y_train": y_train,
            "aux_train": aux_train
        }, data_cache_file)
        print("Adatok sikeresen lementve a cache-be!")

    x_concat = np.concatenate(x_train, axis=0)
    y_concat = np.concatenate(y_train, axis=0)
    aux_concat = np.concatenate(aux_train, axis=0)

    # A preprocessor észre fogja venni, hogy az x_raw már 5 oszlopos, és mindegyiket skálázza!
    preprocessor.fit_transform(x_raw=x_concat, y_raw=y_concat, aux_raw=aux_concat, target_idx=TARGET_IDX)
    #preprocessor.t_max = torch.tensor(200.0, dtype=torch.float32)
    t_unscaled = np.arange(200).reshape(-1, 1)

    x_scaled_list, y_scaled_list, aux_scaled_list, t_list = [], [], [], []
    for i in range(NUM_STATES):
        x_scaled_list.append(preprocessor.x_scaler.transform(x_train[i]))
        y_scaled_list.append(preprocessor.y_scaler.transform(y_train[i]))
        aux_scaled_list.append(preprocessor.aux_scaler.transform(aux_train[i]))
        t_list.append(t_unscaled)

    # ==========================================
    # 2. A NÖVEKVŐ ABLAKOS DATASET ÉS TANÍTÁS
    # ==========================================
    print("Növekvő ablakos adathalmaz létrehozása...")
    dataset = ExpandingWindowDataset(
        x_list=x_scaled_list, y_list=y_scaled_list, t_list=t_list, aux_list=aux_scaled_list,
        min_days=20, max_days=200, step=16, device=device
    )

    train_cfg = EINNTrainConfig(
        device=device, learning_rate=0.001, epochs_per_phase=450, early_stopping_patience=40,
        loss_weights={
            'data_T': 1.0, 'data_F': 1.0,
            'aux': 0.0,
            'ode_T': 10.0, 'ode_F': 10.0,
            'ode_future_T': 0.0, 'ode_future_F': 0.0,
            'mono': 5.0,
            'param': 0.0,
            'kd_target': 1.00, 'kd_emb': 1.0,
            'ic': 5.0, 'ic_F': 0.0
        }
    )

     # {
# 'data_T': 10.0, 'data_F': 10.0,
# 'aux': 2.0,
# 'ode_T': 1.0, 'ode_F': 1.0,
# 'ode_future_T': 0.1, 'ode_future_F': 0.1,
# 'mono': 1.0,
# 'param': 0.01,
# 'kd_target': 0.50, 'kd_emb': 0.5,
# 'ic': 10.0, 'ic_F': 10.0
#} 

#{
        #    'data_T': 8.0, 'data_F': 10.0,
        #    'aux': 1.0,
        #    'ode_T': 5.0, 'ode_F': 7.0,
        #    'ode_future_T': 0.5, 'ode_future_F': 0.5,
        #    'mono': 2.5,
        #    'param': 0.1,
        #    'kd_target': 1.10, 'kd_emb': 1.1,
        #    'ic': 10.0, 'ic_F': 10.0
        #}

    train_cfg.aux_mean = preprocessor.aux_mean.to(device)
    train_cfg.aux_std = preprocessor.aux_std.to(device)
    #train_cfg.t_max = preprocessor.t_max.to(device)

    ic_physical = torch.tensor([0.99, 0.01, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    train_cfg.initial_states = ((ic_physical - train_cfg.aux_mean) / (train_cfg.aux_std + 1e-8)).cpu().tolist()

    # Mivel az adathalmazban X már 5D, a d_x=5 tökéletesen illeszkedik!
    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=5, d_p=4, feature_n_layers=2)
    models = EINNBuilder.build_einn(
        model_config=model_cfg, train_config=train_cfg,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
        model_type="SEIRM", seed=42
    )
    models.ode_model.target_idx = TARGET_IDX

    print("Hálózat betanítása...")
    trainer = EINNTrainer(models=models, config=train_cfg)
    trainer.train(dataset=dataset, epochs=450, batch_size=32)

    # ==========================================
    # 3. AZ 51. ÁLLAM GENERÁLÁSA (SOSEM LÁTOTT ADAT)
    # ==========================================
    print("\nGeneráljuk a sosem látott teszt államot...")
    
    # A teszt államnak is adunk egyedi korrelációs profilokat (Zero-Shot korrelációkra is!)
    test_correlations = {
        'mobility': -0.75,
        'temperature': -0.1,
        'testing': 0.85,
        'stringency': 0.95
    }
    
    generator_test = SyntheticDataGenerator(true_ode_model=true_model, seq_len=200)
    data_test = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
        true_params=[0.45, 0.15, 0.12, 0.04],
        noise_std=0.01,
        correlations=test_correlations
    )

    x_test_scaled = preprocessor.x_scaler.transform(data_test["x_raw"])
    true_i = data_test["true_latent_states"][:, 2]

    # ==========================================
    # 4. INFERENCIA AZ 5 FÁZISRA A TESZT ADATON
    # ==========================================
    peak_day = int(np.argmax(true_i))
    peak_val = true_i[peak_day]
    rise_day = np.where((true_i >= peak_val * 0.2) & (np.arange(200) < peak_day))[0][0]
    decay_day = np.where((true_i <= peak_val * 0.5) & (np.arange(200) > peak_day))[0][0]

    phases = [
        ("1. Kezdeti szakasz (10 nap)", 10),
        (f"2. Felszálló ág ({rise_day}. nap)", rise_day),
        (f"3. Járványcsúcs ({peak_day}. nap)", peak_day),
        (f"4. Lecsengés fele ({decay_day}. nap)", decay_day),
        ("5. Teljes lecsengés (150. nap)", 150)
    ]

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 25))
    fig.suptitle("EINN Általánosítás: Inferencia Sosem Látott Államon (Zero-Shot)", fontsize=20, y=0.98)

    engine = InferenceEngine(models)
    full_t_raw = torch.arange(200, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(device)

    for idx, (phase_name, t_end) in enumerate(phases):
        print(f"Inferencia futtatása: {phase_name}...")

        x_test_slice = torch.tensor(x_test_scaled[:t_end], dtype=torch.float32, device=device).unsqueeze(0)

        states_pred = engine.predict_future(
            historical_x=x_test_slice,
            historical_x_mask=None,
            future_t=full_t_raw,
            z_mean=train_cfg.aux_mean,
            z_std=train_cfg.aux_std,
            #t_max=train_cfg.t_max
        )

        pred_np = states_pred.squeeze(0).cpu().numpy()

        with torch.no_grad():
            future_t_scaled = full_t_raw #/ (train_cfg.t_max + 1e-8)
            e_t_f = models.feature_module(x=x_test_slice, t=future_t_scaled, mask=None)
            emb_np = e_t_f.squeeze(0).cpu().numpy()

        mse_future = mean_squared_error(true_i[t_end:], pred_np[t_end:, 2]) if t_end < 200 else mean_squared_error(
            true_i, pred_np[:, 2])

        ax_pred = axes[idx, 0]
        ax_pred.plot(true_i, label='Valós I (Sosem Látott)', color='black', linewidth=2)
        ax_pred.plot(pred_np[:, 2], label='Prediktált I', color='red', linestyle='--', linewidth=2)
        ax_pred.axvline(x=t_end, color='blue', linestyle=':', linewidth=1.5, label='Bemeneti Ablak Vége')
        ax_pred.fill_betweenx([0, max(true_i) * 1.2], 0, t_end, color='blue', alpha=0.05)
        ax_pred.set_title(f"{phase_name}\nJövőbeli MSE: {mse_future:.2e}")
        ax_pred.set_ylabel("Infected Arány")
        ax_pred.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_pred.legend()

        ax_tsne = axes[idx, 1]
        tsne = TSNE(n_components=2, perplexity=10.0, random_state=42)
        emb_2d = tsne.fit_transform(emb_np)

        ax_tsne.scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(200), cmap='viridis', s=20, edgecolor='k', alpha=0.7)
        ax_tsne.plot(emb_2d[:, 0], emb_2d[:, 1], color='gray', alpha=0.3, linewidth=1)
        ax_tsne.scatter(emb_2d[0, 0], emb_2d[0, 1], color='red', marker='s', s=80, label='Start (0. nap)', zorder=5)
        ax_tsne.scatter(emb_2d[-1, 0], emb_2d[-1, 1], color='black', marker='X', s=100, label='End (200. nap)',
                        zorder=5)
        ax_tsne.scatter(emb_2d[t_end - 1, 0], emb_2d[t_end - 1, 1], color='blue', marker='o', s=80,
                        label=f'Jelen ({t_end}. nap)', zorder=5)
        ax_tsne.set_title(f"t-SNE Látens Tér ({model_cfg.d_e}D -> 2D)")
        ax_tsne.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_tsne.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filepath = os.path.join(output_dir, "generalization_inference.png")
    plt.savefig(filepath, dpi=300)
    print(f"\nA kísérlet lefutott. Ábra: {filepath}")

if __name__ == "__main__":
    run_generalization_experiment()
