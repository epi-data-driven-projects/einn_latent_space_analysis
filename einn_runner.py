import os
import logging
from typing import List, Tuple

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_preprocessor import DataPreprocessor
from einn.data.einn_dataset import EINNDataset
from einn.data.ode_precalibrator import ODEPreCalibrator
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.model.einn_builder import EINNBuilder
from einn.model.interface.einn_models import EINNModels
from einn.model.interface.training_metrics import TrainingMetrics
from einn.ode.base_ode_model import BaseODEModel
from einn.ode.sir_model import SIRModel
from einn.ode.seirm_model import SEIRMModel
from einn.inference.training_post_processor import TrainingPostProcessor
from einn.training.einn_trainer import EINNTrainer


class EINNRunner:
    """
    Központi vezérlő osztály az EINN (Epidemiologically-Informed Neural Network) kutatási
    kísérletek teljes életciklusának futtatására.
    """

    def __init__(self, output_dir: str, model_config: EINNModelConfig, train_config: EINNTrainConfig):
        self.output_dir = output_dir
        self.model_config = model_config
        self.train_config = train_config

        os.makedirs(self.output_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "experiment.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("EINNRunner")

    def run_experiment(
            self,
            true_model: BaseODEModel,
            true_initial_states: List[float],
            true_params: List[float],
            assumed_model: BaseODEModel,
            assumed_d_p: int,
            seq_len: int = 150,
            window_size: int = 15,
            epochs: int = 100,
            batch_size: int = 32,
            observable_state_indices: List[int] = None
    ) -> Tuple[EINNModels, List[TrainingMetrics], DataPreprocessor]:
        self.logger.info("=== 1. FÁZIS: Szintetikus Adatgenerálás (Valóság szimulálása) ===")
        # Nagyon enyhe zaj a baseline-hoz (0.01), hogy biztosan megtalálja a fizikai összefüggéseket
        generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=seq_len)
        data_dict = generator.generate(
            initial_states=true_initial_states,
            true_params=true_params,
            noise_std=0.02,
            observable_state_indices=observable_state_indices
        )
        x_raw = data_dict["x_raw"]
        y_raw = data_dict["y_raw"]
        self.model_config.d_x = x_raw.shape[1]
        self.logger.info(f"Adatok generálva. x_raw forma: {x_raw.shape}, y_raw forma: {y_raw.shape}")

        self.logger.info("=== 2. FÁZIS: ODE Előkalibráció (Kutatói feltevés illesztése) ===")

        calibrator = ODEPreCalibrator(assumed_ode_model=assumed_model, d_p=assumed_d_p, device=self.train_config.device)

        # Mivel SEIRM modellt (5 rekesz) feltételezünk, egy 5 elemű kezdőállapotot adunk be mankónak!
        # Tudjuk, hogy 99% fogékony és 1% fertőzött van az elején.
        assumed_initial_states = [0.99, 0.01, 0.0, 0.0, 0.0] if assumed_model.d_s == 5 else [0.99, 0.01, 0.0]

        aux_raw = calibrator.calibrate(
            y_raw=y_raw,
            initial_states=assumed_initial_states,  # <--- ÚJ PARAMÉTER!
            epochs=500,
            lr=0.01
        )
        self.logger.info(f"Előkalibráció kész. aux_raw forma: {aux_raw.shape}")
        self.logger.info("=== 3. FÁZIS: Adatok Normalizálása (Z-score Skálázás) ===")
        preprocessor = DataPreprocessor()
        batch_dict = preprocessor.fit_transform(
            x_raw=x_raw,
            y_raw=y_raw,
            aux_raw=aux_raw,
            target_idx=assumed_model.target_idx  # <--- EZT ADD HOZZÁ!
        )

        self.train_config.aux_mean = preprocessor.aux_mean
        self.train_config.aux_std = preprocessor.aux_std
        self.train_config.t_max = preprocessor.t_max

        # === IC-LOSS SKÁLÁZÁS JAVÍTÁSA ===
        # Az assumed_initial_states fizikai érték. A hálózat viszont Z-score térben tanul.
        # Normalizálnunk kell a kezdőértékeket a preprocessor statisztikáival!
        import torch

        # PyTorch Tenzort csinálunk belőle, ugyanazon az eszközön (CPU/GPU), ahol az aux_mean is van
        ic_physical = torch.tensor(assumed_initial_states, dtype=torch.float32, device=preprocessor.aux_mean.device)

        # Végrehajtjuk a Z-score normalizálást a tenzorokon
        ic_normalized = (ic_physical - preprocessor.aux_mean) / (preprocessor.aux_std + 1e-8)

        # Visszaalakítjuk Python listává a konfigurációhoz
        self.train_config.initial_states = ic_normalized.cpu().tolist()
        self.logger.info(f"Normalizált IC értékek beállítva: {self.train_config.initial_states}")
        # =================================

        dataset = EINNDataset(
            x=batch_dict["x"],
            y=batch_dict["y"],
            t=batch_dict["t"],
            aux_targets=batch_dict["aux_targets"],
            window_size=window_size,
            device=self.train_config.device
        )
        self.logger.info(f"Dataset létrehozva. Csúszóablakok száma: {len(dataset)}")

        self.logger.info("=== 4. FÁZIS: EINN Modellek Inicializálása ===")

        # Dinamikus paraméterek az ODE inicializálásához
        if assumed_d_p == 2:
            model_t = "SIR"
            init_params = {"beta": 0.3, "gamma": 0.1}
        else:
            model_t = "SEIRM"
            init_params = {"beta": 0.3, "alpha": 0.2, "gamma": 0.1, "mu": 0.05}

        # Buildeljük a neurális hálózatokat az EINNBuilder-rel dinamikusan
        models = EINNBuilder.build_einn(
            model_config=self.model_config,
            train_config=self.train_config,
            param_calibration=init_params,
            model_type=model_t,
            seed=42
        )

        # A target_idx biztonságos átadása a kutatói feltevésből
        models.ode_model.target_idx = assumed_model.target_idx
        self.logger.info(f"Modellek sikeresen inicializálva ({model_t} ODE motorral).")

        self.logger.info("=== 5. FÁZIS: EINN Hálózat Tanítása ===")
        trainer = EINNTrainer(models=models, config=self.train_config)
        metrics = trainer.train(dataset=dataset, epochs=epochs, batch_size=batch_size)

        self.logger.info("=== 6. FÁZIS: Utófeldolgozás és Mentés ===")
        post_processor = TrainingPostProcessor(output_dir=self.output_dir)
        post_processor.save_models(models=models, prefix="einn_experiment")
        post_processor.save_metrics(metrics=metrics)
        post_processor.plot_loss_curves(metrics=metrics, filename="loss_curves.png")

        # ---- Ezt a 7. Fázist kell visszatenni! ----
        self.logger.info("=== 7. FÁZIS: Predikció és Valóság Összehasonlítása ===")
        from einn.inference.inference_engine import InferenceEngine
        engine = InferenceEngine(models)

        full_x = batch_dict["x"].unsqueeze(0).to(self.train_config.device)
        full_t = batch_dict["t"].unsqueeze(0).to(self.train_config.device)

        predicted_states = engine.predict_future(
            historical_x=full_x,
            historical_x_mask=None,
            future_t=full_t,
            z_mean=preprocessor.aux_mean,
            z_std=preprocessor.aux_std,
            t_max=self.train_config.t_max
        )

        pred_np = predicted_states.squeeze(0).cpu().numpy()
        true_np = data_dict["true_latent_states"]

        post_processor.plot_true_vs_predicted(true_states=true_np, predicted_states=pred_np)
        # ---------------------------------------------

        self.logger.info(f"Kísérlet befejezve! Eredmények a(z) {self.output_dir} mappában.")
        return models, metrics, preprocessor

    def run_generalization_experiment(
            self,
            true_model: BaseODEModel,
            assumed_model: BaseODEModel,
            training_scenarios: List[dict],
            test_scenario: dict,
            epochs: int = 150,
            window_size: int = 14,
            batch_size: int = 32,
            observable_state_indices: List[int] = None
    ) -> Tuple[EINNModels, List[TrainingMetrics]]:

        self.logger.info("=== ÁLTALÁNOSÍTÁSI KÍSÉRLET: Több trajektória generálása ===")

        all_x_windows, all_y_windows, all_t_windows, all_aux_windows = [], [], [], []
        preprocessor = DataPreprocessor()

        # 1. Tanuló adatok (Több eltérő járvány) legenerálása
        for i, scenario in enumerate(training_scenarios):
            self.logger.info(f"Generálás - {i + 1}. Járvány Szcenárió: {scenario['params']}")

            generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=scenario['seq_len'])
            data_dict = generator.generate(
                initial_states=scenario['init_states'],
                true_params=scenario['params'],
                noise_std=0.01,
                observable_state_indices=observable_state_indices
            )

            # Kalibráció az aktuális szcenárióra
            calibrator = ODEPreCalibrator(assumed_ode_model=assumed_model, d_p=4, device=self.train_config.device)
            assumed_initial_states = [0.99, 0.01, 0.0, 0.0, 0.0] if assumed_model.d_s == 5 else [0.99, 0.01, 0.0]
            aux_raw = calibrator.calibrate(data_dict["y_raw"], initial_states=assumed_initial_states, epochs=300)

            # Skálázás (az első szcenárió statisztikáit használjuk referenciaként)
            if i == 0:
                batch_dict = preprocessor.fit_transform(data_dict["x_raw"], data_dict["y_raw"], aux_raw,
                                                        assumed_model.target_idx)
                self.train_config.aux_mean = preprocessor.aux_mean
                self.train_config.aux_std = preprocessor.aux_std
                self.train_config.t_max = preprocessor.t_max
                self.model_config.d_x = data_dict["x_raw"].shape[1]
            else:
                batch_dict = preprocessor.transform(data_dict["x_raw"], data_dict["y_raw"], aux_raw)

            # Csúszóablakok létrehozása KIZÁRÓLAG az adott szcenárión belül!
            dataset = EINNDataset(batch_dict["x"], batch_dict["y"], batch_dict["t"], batch_dict["aux_targets"],
                                  window_size, self.train_config.device)

            # Kimentjük az ablakokat
            for idx in range(len(dataset)):
                item = dataset[idx]
                all_x_windows.append(item["x"])
                all_y_windows.append(item["y"])
                all_t_windows.append(item["t"])
                all_aux_windows.append(item["aux_targets"])

        # 2. Gigantikus Közös Dataset létrehozása
        import torch
        from torch.utils.data import TensorDataset

        combined_dataset = TensorDataset(
            torch.stack(all_x_windows),
            torch.stack(all_y_windows),
            torch.stack(all_t_windows),
            torch.stack(all_aux_windows)
        )

        # Az EINNTrainer egy sima Dataset-et vár, aminek dict a kimenete, ezt egy kis wrapperrel megoldjuk
        class CombinedDictDataset(torch.utils.data.Dataset):
            def __init__(self, tensor_dataset): self.ds = tensor_dataset

            def __len__(self): return len(self.ds)

            def __getitem__(self, idx):
                x, y, t, aux = self.ds[idx]
                return {"x": x, "y": y, "t": t, "aux_targets": aux}

        final_dataset = CombinedDictDataset(combined_dataset)
        self.logger.info(f"Összesített betanító ablakok száma: {len(final_dataset)}")

        # 3. Modellek inicializálása és tanítása az óriási adathalmazon
        init_params = {"beta": 0.3, "alpha": 0.2, "gamma": 0.1, "mu": 0.05}
        models = EINNBuilder.build_einn(self.model_config, self.train_config, init_params, "SEIRM", seed=42)
        models.ode_model.target_idx = assumed_model.target_idx

        trainer = EINNTrainer(models=models, config=self.train_config)
        metrics = trainer.train(dataset=final_dataset, epochs=epochs, batch_size=batch_size)

        self.logger.info("=== TANÍTÁS KÉSZ. Jöhet a sosem látott TESZT szcenárió kiértékelése! ===")
        # Teszt adatok generálása
        test_generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=test_scenario['seq_len'])
        test_data_dict = test_generator.generate(
            initial_states=test_scenario['init_states'],
            true_params=test_scenario['params'],
            noise_std=0.01,  # Itt is lehet zaj, hogy reális legyen az X bemenet
            observable_state_indices=observable_state_indices
        )

        # Teszt Mankó kalibrálása (az LBFGS-nek meg kell találnia a teszt görbe arányait)
        test_calibrator = ODEPreCalibrator(assumed_ode_model=assumed_model, d_p=4, device=self.train_config.device)
        test_assumed_init = [0.99, 0.01, 0.0, 0.0, 0.0] if assumed_model.d_s == 5 else [0.99, 0.01, 0.0]
        test_aux_raw = test_calibrator.calibrate(test_data_dict["y_raw"], initial_states=test_assumed_init, epochs=300)

        # FIGYELEM: Itt már csak .transform()! A hálózat a tanulóhalmaz átlagához/szórásához van szokva.
        test_batch_dict = preprocessor.transform(test_data_dict["x_raw"], test_data_dict["y_raw"], test_aux_raw)

        # 5. FÁZIS: Predikció az Inference Engine-nel
        from einn.inference.inference_engine import InferenceEngine
        engine = InferenceEngine(models)

        full_x_test = test_batch_dict["x"].unsqueeze(0).to(self.train_config.device)
        full_t_test = test_batch_dict["t"].unsqueeze(0).to(self.train_config.device)

        predicted_states = engine.predict_future(
            historical_x=full_x_test,
            historical_x_mask=None,
            future_t=full_t_test,
            z_mean=preprocessor.aux_mean,
            z_std=preprocessor.aux_std,
            t_max=self.train_config.t_max
        )

        pred_np = predicted_states.squeeze(0).cpu().numpy()
        true_np = test_data_dict["true_latent_states"]

        # 6. FÁZIS: Eredmények kimentése (Loss görbék a tanításról, true_vs_pred a TESZT adatról)
        post_processor = TrainingPostProcessor(output_dir=self.output_dir)
        post_processor.save_models(models=models, prefix="einn_generalization")
        post_processor.save_metrics(metrics=metrics)
        post_processor.plot_loss_curves(metrics=metrics, filename="loss_curves.png")
        post_processor.plot_true_vs_predicted(true_states=true_np, predicted_states=pred_np)

        self.logger.info(f"Általánosítási kísérlet befejezve! Eredmények a(z) {self.output_dir} mappában.")
        return models, metrics


def run_generalization_test():
    model_cfg, train_cfg = get_standard_configs()
    runner = EINNRunner(output_dir="./experiment_results/03_generalization", model_config=model_cfg,
                        train_config=train_cfg)

    # 3 teljesen eltérő járvány (Gyors, Lassú, Halálos)
    training_scenarios = [
        {'params': [0.5, 0.2, 0.1, 0.02], 'init_states': [0.99, 0.01, 0.0, 0.0, 0.0], 'seq_len': 200},
        {'params': [0.25, 0.1, 0.05, 0.01], 'init_states': [0.99, 0.01, 0.0, 0.0, 0.0], 'seq_len': 200},
        {'params': [0.4, 0.2, 0.1, 0.1], 'init_states': [0.99, 0.01, 0.0, 0.0, 0.0], 'seq_len': 200}
    ]

    # Egy 4., ismeretlen járvány a teszteléshez
    test_scenario = {'params': [0.35, 0.15, 0.08, 0.05], 'init_states': [0.99, 0.01, 0.0, 0.0, 0.0], 'seq_len': 200}

    runner.run_generalization_experiment(
        true_model=SEIRMModel(target_idx=4),
        assumed_model=SEIRMModel(target_idx=4),
        training_scenarios=training_scenarios,
        test_scenario=test_scenario,
        window_size=14,  # Itt nyugodtan mehet a rövid ablak!
        epochs=150,
        batch_size=32,
        observable_state_indices=[4]
    )


# ==========================================
# KÍSÉRLET VEZÉRLŐ FÜGGVÉNYEK
# ==========================================

def get_standard_configs() -> Tuple[EINNModelConfig, EINNTrainConfig]:
    """Minden kísérlet számára egységes hiperparamétereket biztosít."""
    model_cfg = EINNModelConfig(
        d_x=5, d_e=32, d_s=5, d_p=4, feature_n_layers=2
    )

    train_cfg = EINNTrainConfig(
        device='cpu',
        learning_rate=0.002,
        epochs_per_phase=300,
        early_stopping_patience=40,
        reps={'phase_1': 1, 'phase_2': 1, 'phase_3': 1, 'phase_4': 1},
        loss_weights={
            'data_T': 5.0, 'data_F': 5.0,
            'aux': 5.0,  # VISSZAÁLLÍTVA: Ez tartja egyben a globális struktúrát!
            'ode_T': 2.0, 'ode_F': 2.0,
            'ode_future_T': 0.1, 'ode_future_F': 0.1,
            'mono': 2.0, 'param': 0.1,
            'ic': 5.0,   # DRASZTIKUSAN CSÖKKENTVE: Csak finom iránymutatás, elkerüli a robbanást.
            'kd_target': 1.0, 'kd_emb': 1.0,
        },
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0]
    )

    return model_cfg, train_cfg


def run_seirm_sanity_check():
    """KONTROLL: SEIRM valóság illesztése SEIRM feltevésre (Helyes modellstruktúra)"""
    print("\n" + "=" * 50)
    print("INDÍTÁS: KONTROLL KÍSÉRLET (SEIRM -> SEIRM)")
    print("=" * 50)

    model_cfg, train_cfg = get_standard_configs()
    runner = EINNRunner(output_dir="./experiment_results/01_sanity_seirm", model_config=model_cfg,
                        train_config=train_cfg)

    # Valóság és feltevés MEGEGYEZIK. Célváltozó: Mortalitás (M -> 4. index)
    runner.run_experiment(
        true_model=SEIRMModel(target_idx=4),
        true_initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
        true_params=[0.4, 0.2, 0.1, 0.05],
        assumed_model=SEIRMModel(target_idx=4),
        assumed_d_p=4,
        seq_len=200,
        window_size=200,
        epochs=200,
        batch_size=1,

        # Bemenetként is az M görbét (4) adjuk át a 4 exogén adat mellett
        observable_state_indices=[4]
    )


def run_structural_error_experiment():
    """TESZT: SIR valóság illesztése SEIRM feltevésre (Strukturális Modellhiba)"""
    print("\n" + "=" * 50)
    print("INDÍTÁS: STRUKTURÁLIS HIBA KÍSÉRLET (SIR -> SEIRM)")
    print("=" * 50)

    model_cfg, train_cfg = get_standard_configs()
    runner = EINNRunner(output_dir="./experiment_results/02_experiment_sir_seirm", model_config=model_cfg,
                        train_config=train_cfg)

    # Valóság (SIR) Eltér a feltevéstől (SEIRM)
    runner.run_experiment(
        true_model=SIRModel(target_idx=1),  # Cél: I (1)
        true_initial_states=[0.99, 0.01, 0.0],
        true_params=[0.3, 0.1],
        assumed_model=SEIRMModel(target_idx=2),  # Cél: I (2)
        assumed_d_p=4,
        seq_len=200,
        window_size=14,  # Ugyanaz a lokális ablak!
        epochs=150,
        batch_size=16
    )


if __name__ == "__main__":
    # Válassz a futtatási módok közül!
    RUN_MODE = "GENERALIZATION"  # Lehetőségek: "SANITY", "EXPERIMENT", "GENERALIZATION", "ALL"

    if RUN_MODE in ["SANITY", "ALL"]:
        run_seirm_sanity_check()

    if RUN_MODE in ["EXPERIMENT", "ALL"]:
        run_structural_error_experiment()

    if RUN_MODE in ["GENERALIZATION", "ALL"]:
        # Mielőtt ezt futtatod, győződj meg róla, hogy a korrelált adatok kódja
        # (SyntheticDataGenerator) már benne van a rendszeredben!
        run_generalization_test()
