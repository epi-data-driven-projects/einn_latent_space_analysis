import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import paired_cosine_distances

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
NUM_STATES = 20  # Ezt később egyszerűen átírhatod 50-re!


def seed_everything(seed: int = 42):
    """
    Globálisan rögzíti az összes véletlenszám-generátort a teljes determinizmus érdekében.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Multi-GPU esetén
    
    # A CuDNN determinizmus kikényszerítése (ez elengedhetetlen a GRU/LSTM hálózatoknál)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    if TARGET_IDX == 4:
        target_name = "Mortality"
    if TARGET_IDX == 2:
        target_name = "Infected"
    if TARGET_IDX == 1:
        target_name = "Exposed"
    
    data_cache_file = os.path.join(output_dir, f"{NUM_STATES}_cached_seirm_data_target_{TARGET_IDX}.pt")
    
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
        
        if len(cached_data["x_train"]) >= NUM_STATES:
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

            # Dinamikus korrelációk minden államhoz
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
                noise_std=0.02,
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

    preprocessor.fit_transform(x_raw=x_concat, y_raw=y_concat, aux_raw=aux_concat, target_idx=TARGET_IDX)
    # JAVÍTÁS: A t_min explicit hozzáadása a biztonság kedvéért!
    preprocessor.t_min = torch.tensor(0.0, dtype=torch.float32)
    preprocessor.t_max = torch.tensor(200.0, dtype=torch.float32)
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
        min_days=20, max_days=200, step=20, device=device
    )

    train_cfg = EINNTrainConfig(
        device=device, learning_rate=0.001, epochs_per_phase=450, early_stopping_patience=5,
        loss_weights={
            'data_T': 10.0, 'data_F': 10.0,
            'aux': 4.0,
            'ode_T': 8.0, 'ode_F': 8.0,
            'ode_future_T': 1.0, 'ode_future_F': 1.0,
            'mono': 3.0,
            'param': 0.1,
            'kd_target': 2.0, 'kd_emb': 2.0,
            'ic': 10.0, 'ic_F': 10.0
        }
    )

    train_cfg.aux_mean = preprocessor.aux_mean.to(device)
    train_cfg.aux_std = preprocessor.aux_std.to(device)
    train_cfg.t_min = preprocessor.t_min.to(device)
    train_cfg.t_max = preprocessor.t_max.to(device)
    train_cfg.reps = {
        'phase_1': 100,
        'phase_2': 100,
        'phase_3': 100,
        'phase_4': 100
    }

    ic_physical = torch.tensor([0.99, 0.01, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    train_cfg.initial_states = ((ic_physical - train_cfg.aux_mean) / (train_cfg.aux_std + 1e-8)).cpu().tolist()

    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=5, d_p=4, feature_n_layers=2)
    models = EINNBuilder.build_einn(
        model_config=model_cfg, train_config=train_cfg,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
        model_type="SEIRM", seed=42
    )
    models.ode_model.target_idx = TARGET_IDX

    print("Hálózat betanítása...")
    trainer = EINNTrainer(models=models, config=train_cfg)
    trainer.train(dataset=dataset, epochs=20, batch_size=32)

    # ==========================================
    # 3. AZ 51. ÁLLAM GENERÁLÁSA (SOSEM LÁTOTT ADAT)
    # ==========================================
    print("\nGeneráljuk a sosem látott teszt államot...")
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

    true_target = data_test["true_latent_states"][:, TARGET_IDX]

    # ==========================================
    # 4. INFERENCIA & 4x6 GRID LÁTENS TÉR ANALÍZIS
    # ==========================================
    peak_day = int(np.argmax(true_i))
    peak_val = true_i[peak_day]
    rise_day = np.where((true_i >= peak_val * 0.2) & (np.arange(200) < peak_day))[0][0]
    decay_day = np.where((true_i <= peak_val * 0.5) & (np.arange(200) > peak_day))[0][0]

    phases = [
        (f"1. Felszálló ág ({rise_day}. nap)", rise_day),
        (f"2. Járványcsúcs ({peak_day}. nap)", peak_day),
        (f"3. Lecsengés ({decay_day}. nap)", decay_day),
        ("4. Járvány vége (150. nap)", 150)
    ]

    # Óriási 4x6-os rács beállítása
    fig = plt.figure(figsize=(32, 20))
    fig.suptitle(f"EINN Generalizáció: Inferencia ({target_name}) és Látens Tér Evolúció", fontsize=24, y=0.98)

    engine = InferenceEngine(models)
    full_t_raw = torch.arange(200, dtype=torch.float32).unsqueeze(1).unsqueeze(0).to(device)

    # JAVÍTÁS: A Time Module hívása előtt teljes Min-Max skálázás történik az időn
    with torch.no_grad():
        t_range = train_cfg.t_max - train_cfg.t_min + 1e-8
        future_t_scaled = (full_t_raw - train_cfg.t_min) / t_range
        e_t_t_tensor = models.time_module(t=future_t_scaled)
        emb_np_t = e_t_t_tensor.squeeze(0).cpu().numpy()

    for idx, (phase_name, t_end) in enumerate(phases):
        print(f"Analízis futtatása: {phase_name}...")
        x_test_slice = torch.tensor(x_test_scaled[:t_end], dtype=torch.float32, device=device).unsqueeze(0)

        # 1. Predikció (JAVÍTÁS: t_min átadása a InferenceEngine-nek)
        states_pred = engine.predict_future(
            historical_x=x_test_slice, historical_x_mask=None, future_t=full_t_raw,
            z_mean=train_cfg.aux_mean, z_std=train_cfg.aux_std, 
            t_min=train_cfg.t_min, 
            t_max=train_cfg.t_max
        )
        pred_np = states_pred.squeeze(0).cpu().numpy()

        # 2. Látens tér
        with torch.no_grad():
            e_t_f = models.feature_module(x=x_test_slice, t=future_t_scaled, mask=None)
            emb_np_f = e_t_f.squeeze(0).cpu().numpy()

        mse_future = mean_squared_error(true_target[t_end:], pred_np[t_end:, TARGET_IDX]) if t_end < 200 else mean_squared_error(true_target, pred_np[:, TARGET_IDX])

        # --- A 6 PANEL HOZZÁADÁSA ---
        ax_pred    = fig.add_subplot(4, 6, idx * 6 + 1)
        ax_tsne    = fig.add_subplot(4, 6, idx * 6 + 2)
        ax_pca_var = fig.add_subplot(4, 6, idx * 6 + 3)
        ax_pca_2d  = fig.add_subplot(4, 6, idx * 6 + 4)
        ax_pca_3d  = fig.add_subplot(4, 6, idx * 6 + 5, projection='3d') # A 3D-s panel!
        ax_cos     = fig.add_subplot(4, 6, idx * 6 + 6)

        # 1. Predikciós Görbe
        ax_pred.plot(true_target, label=f'Valós {target_name}', color='black', linewidth=2)
        ax_pred.plot(pred_np[:, TARGET_IDX], label=f'Prediktált', color='red', linestyle='--', linewidth=2)
        ax_pred.axvline(x=t_end, color='blue', linestyle=':', linewidth=1.5, label='Ablak Vége')
        ax_pred.axvspan(0, t_end, color='blue', alpha=0.05)
        ax_pred.set_title(f"{phase_name}")
        ax_pred.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: ax_pred.legend()

        # 2. t-SNE
        tsne = TSNE(n_components=2, perplexity=10.0, random_state=42)
        emb_tsne = tsne.fit_transform(emb_np_f)
        ax_tsne.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=np.arange(200), cmap='viridis', s=20, edgecolor='k', alpha=0.7)
        ax_tsne.plot(emb_tsne[:, 0], emb_tsne[:, 1], color='gray', alpha=0.3, linewidth=1)
        ax_tsne.scatter(emb_tsne[0, 0], emb_tsne[0, 1], color='red', marker='s', s=80, label='Start (0. nap)', zorder=5)
        ax_tsne.scatter(emb_tsne[-1, 0], emb_tsne[-1, 1], color='black', marker='X', s=100, label='End', zorder=5)
        ax_tsne.scatter(emb_tsne[t_end - 1, 0], emb_tsne[t_end - 1, 1], color='blue', marker='o', s=80, label='Jelen', zorder=5)
        ax_tsne.set_title(f"t-SNE ({model_cfg.d_e}D -> 2D)")
        ax_tsne.grid(True, linestyle='--', alpha=0.5)

        # 3. PCA Variancia
        pca = PCA()
        pca.fit(emb_np_f)
        cum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_comp = np.argmax(cum_variance >= 0.99) + 1
        ax_pca_var.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='-', color='teal')
        ax_pca_var.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax_pca_var.axhline(y=0.99, color='g', linestyle='--', label='99%')
        ax_pca_var.set_title(f"PCA Variancia (99%-hoz: {n_comp} dim)")
        ax_pca_var.set_xlabel("Főkomponensek")
        ax_pca_var.grid(True, linestyle='--', alpha=0.6)

        # PCA Dimenziócsökkentés 3D-be
        pca_coords = PCA(n_components=3).fit_transform(emb_np_f)

        # 4. PCA 2D
        ax_pca_2d.scatter(pca_coords[:, 0], pca_coords[:, 1], c=np.arange(200), cmap='viridis', s=20, edgecolor='k', alpha=0.7)
        ax_pca_2d.plot(pca_coords[:, 0], pca_coords[:, 1], color='gray', alpha=0.3, linewidth=1)
        ax_pca_2d.scatter(pca_coords[0, 0], pca_coords[0, 1], color='red', marker='s', s=80, label='Start', zorder=5)
        ax_pca_2d.scatter(pca_coords[-1, 0], pca_coords[-1, 1], color='black', marker='X', s=100, label='End', zorder=5)
        ax_pca_2d.scatter(pca_coords[t_end - 1, 0], pca_coords[t_end - 1, 1], color='blue', marker='o', s=80, label='Jelen', zorder=5)
        ax_pca_2d.set_title("PCA Látens Tér (2D)")
        ax_pca_2d.grid(True, linestyle='--', alpha=0.5)

        # 5. PCA 3D
        ax_pca_3d.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2], c=np.arange(200), cmap='viridis', s=20, edgecolor='k', alpha=0.7)
        ax_pca_3d.plot(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2], color='gray', alpha=0.3, linewidth=1)
        ax_pca_3d.scatter(pca_coords[0, 0], pca_coords[0, 1], pca_coords[0, 2], color='red', marker='s', s=80, zorder=5)
        ax_pca_3d.scatter(pca_coords[-1, 0], pca_coords[-1, 1], pca_coords[-1, 2], color='black', marker='X', s=100, zorder=5)
        ax_pca_3d.scatter(pca_coords[t_end - 1, 0], pca_coords[t_end - 1, 1], pca_coords[t_end - 1, 2], color='blue', marker='o', s=80, zorder=5)
        ax_pca_3d.set_title("PCA Látens Tér (3D)")
        ax_pca_3d.xaxis.pane.fill = False
        ax_pca_3d.yaxis.pane.fill = False
        ax_pca_3d.zaxis.pane.fill = False

        # 6. Koszinusz Hasonlóság & Procrustes
        mtx1, mtx2, disparity = procrustes(emb_np_t, emb_np_f)
        cos_sim = 1 - paired_cosine_distances(emb_np_t, emb_np_f)
        ax_cos.plot(cos_sim, color='purple', linewidth=2)
        ax_cos.set_title(f"Time vs Feature (Procrustes: {disparity:.4f})")
        ax_cos.set_xlabel("t")
        ax_cos.set_ylim([-1.1, 1.1])
        ax_cos.grid(True, linestyle='--', alpha=0.6)

        sensitivity_path = os.path.join(output_dir, f"input_sensitivity_target_{TARGET_IDX}.png")

        # 4x6 ábra mentése
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        filepath = os.path.join(output_dir, f"generalization_full_analysis_target_{TARGET_IDX}.png")
        fig.savefig(filepath, dpi=300)
        
        # JAVÍTÁS: Eltört string formázás javítása
        print(f"Kész! Ábrák mentve:\n{filepath}\n{sensitivity_path}")

if __name__ == "__main__":
    seed_everything(seed=42)
    run_generalization_experiment()
