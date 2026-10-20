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


class ExpandingWindowDataset(EINNDataset):
    """
    Kiképzi a modellt arra, hogy [0:t] hosszúságú (paddolt) bemenetből
    kitalálja a teljes 200 napos görbét. Leszármazik az EINNDataset-ből, hogy a Trainer elfogadja.
    """

    def __init__(self, x_list, y_list, t_list, aux_list, min_days=10, max_days=200, step=10, device='cpu'):
        # Kihagyjuk az ősosztály inicializálását, mert saját adatszerkezetet építünk
        self.device = torch.device(device)
        self.samples = []

        for x, y, t, aux in zip(x_list, y_list, t_list, aux_list):
            for current_days in range(min_days, max_days + 1, step):
                # Bemenet paddingelése 200-ra (hogy a batch méretek egyezzenek)
                x_padded = torch.zeros((max_days, x.shape[1]), device=device)
                x_padded[:current_days] = torch.tensor(x[:current_days], dtype=torch.float32, device=device)

                # Maszk: 0 = Valós adat, 1 = Padding (Jövő, amit a hálózat nem láthat)
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
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # ==========================================
    # 1. 50 KÜLÖNBÖZŐ ÁLLAM GENERÁLÁSA (TRAIN)
    # ==========================================
    print("50 különböző paraméterezésű SEIRM állam generálása és kalibrálása...")
    true_model = SEIRMModel(target_idx=4)
    calibrator_model = SEIRMModel(target_idx=4)
    calibrator = ODEPreCalibrator(assumed_ode_model=calibrator_model, d_p=4, device=device)
    preprocessor = DataPreprocessor()

    x_train, y_train, aux_train = [], [], []

    for i in range(2):
        # Véletlenszerű fizikai paraméterek
        beta = np.random.uniform(0.3, 0.6)
        alpha = np.random.uniform(0.1, 0.3)
        gamma = np.random.uniform(0.05, 0.15)
        mu = np.random.uniform(0.01, 0.08)

        generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=200)
        data = generator.generate(
            initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
            true_params=[beta, alpha, gamma, mu],
            noise_std=0.01
        )

        # Mankó kalibrálása minden államhoz külön (LBFGS villámgyors lesz)
        aux = calibrator.calibrate(y_raw=data["y_raw"], initial_states=[0.99, 0.01, 0.0, 0.0, 0.0], epochs=20, lr=0.1)

        x_train.append(data["x_raw"])
        y_train.append(data["y_raw"])

        # TRÜKK alkalmazása a kalibrált mankóra: a target_idx (Mortalitás) oszlop felülírása a zajos adatokkal
        aux[:, 4] = data["y_raw"].ravel()
        aux_train.append(aux)

    # Preprocessor illesztése a teljes, összevont adathalmazra (hogy egy globális skálát kapjunk)
    x_concat = np.concatenate(x_train, axis=0)
    y_concat = np.concatenate(y_train, axis=0)
    aux_concat = np.concatenate(aux_train, axis=0)

    preprocessor.fit_transform(x_raw=x_concat, y_raw=y_concat, aux_raw=aux_concat, target_idx=4)
    preprocessor.t_max = torch.tensor(200.0, dtype=torch.float32)
    t_unscaled = np.arange(200).reshape(-1, 1)  # [200, 1]

    # Skálázás alkalmazása a különálló államokra közvetlenül a Scikit-Learn scaler-ekkel
    x_scaled_list, y_scaled_list, aux_scaled_list, t_list = [], [], [], []
    for i in range(2):
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
        min_days=20, max_days=200, step=10, device=device
    )

    train_cfg = EINNTrainConfig(
        device=device, learning_rate=0.002, epochs_per_phase=150, early_stopping_patience=20,
        loss_weights={
            'ic': 100.0, 'ic_F': 100.0,
            'data_T': 10.0, 'data_F': 10.0, 'aux': 10.0,
            'ode_T': 5.0, 'ode_F': 5.0, 'ode_future_T': 1.0, 'ode_future_F': 1.0,
            'mono': 2.0, 'param': 0.1, 'kd_target': 1.0, 'kd_emb': 1.0,
        }
    )

    train_cfg.aux_mean = preprocessor.aux_mean.to(device)
    train_cfg.aux_std = preprocessor.aux_std.to(device)
    train_cfg.t_max = preprocessor.t_max.to(device)

    ic_physical = torch.tensor([0.99, 0.01, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    train_cfg.initial_states = ((ic_physical - train_cfg.aux_mean) / (train_cfg.aux_std + 1e-8)).cpu().tolist()

    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=5, d_p=4, feature_n_layers=2)
    models = EINNBuilder.build_einn(
        model_config=model_cfg, train_config=train_cfg,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
        model_type="SEIRM", seed=42
    )
    models.ode_model.target_idx = 4

    print("Hálózat betanítása...")
    trainer = EINNTrainer(models=models, config=train_cfg)
    trainer.train(dataset=dataset, epochs=150, batch_size=64)

    # ==========================================
    # 3. AZ 51. ÁLLAM GENERÁLÁSA (SOSEM LÁTOTT ADAT)
    # ==========================================
    print("\nGeneráljuk az 51., sosem látott teszt államot...")
    generator_test = SyntheticDataGenerator(true_ode_model=true_model, seq_len=200)
    data_test = generator_test.generate(
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
        true_params=[0.45, 0.15, 0.12, 0.04],  # Új paraméterkombináció!
        noise_std=0.01
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
        ("1. Kezdeti szakasz (20 nap)", 20),
        (f"2. Felszálló ág ({rise_day}. nap)", rise_day),
        (f"3. Járványcsúcs ({peak_day}. nap)", peak_day),
        (f"4. Lecsengés fele ({decay_day}. nap)", decay_day),
        ("5. Teljes lecsengés (195. nap)", 195)
    ]

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 25))
    fig.suptitle("EINN Általánosítás: Inferencia Sosem Látott Államon (Zero-Shot)", fontsize=20, y=0.98)

    engine = InferenceEngine(models)
    full_t_raw = torch.arange(200, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(device)

    for idx, (phase_name, t_end) in enumerate(phases):
        print(f"Inferencia futtatása: {phase_name}...")

        # PADDOLT BEMENET KÉSZÍTÉSE AZ INFERENCIÁHOZ
        x_test_padded = torch.zeros((1, 200, x_test_scaled.shape[1]), device=device)
        x_test_padded[0, :t_end, :] = torch.tensor(x_test_scaled[:t_end], dtype=torch.float32, device=device)

        mask_test = torch.ones((1, 200), dtype=torch.float32, device=device)
        mask_test[0, :t_end] = 0.0

        # Inferencia a dedikált motorral! (Az InferenceEngine gondoskodik az eval módról és a gradiens blokkolásáról)
        states_pred = engine.predict_future(
            historical_x=x_test_padded,
            historical_x_mask=mask_test,
            future_t=full_t_raw,
            z_mean=train_cfg.aux_mean,
            z_std=train_cfg.aux_std,
            t_max=train_cfg.t_max
        )

        pred_np = states_pred.squeeze(0).cpu().numpy()

        # Kinyerjük a Látens Tér Trajektóriáját a t-SNE-hez (mivel az InferenceEngine csak a fizikai állapotot adja vissza)
        with torch.no_grad():
            future_t_scaled = full_t_raw / (train_cfg.t_max + 1e-8)
            e_t_f = models.feature_module(x=x_test_padded, t=future_t_scaled, mask=mask_test)
            emb_np = e_t_f.squeeze(0).cpu().numpy()

        # --- Ábrázolás ---
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
        tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
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
#    torch.backends.cudnn.enabled = False
    run_generalization_experiment()