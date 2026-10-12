import os
import logging
import numpy as np
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
            batch_size: int = 32
    ) -> Tuple[EINNModels, List[TrainingMetrics], DataPreprocessor]:
        self.logger.info("=== 1. FÁZIS: Szintetikus Adatgenerálás (Valóság szimulálása) ===")
        # Nagyon enyhe zaj a baseline-hoz (0.01), hogy biztosan megtalálja a fizikai összefüggéseket
        generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=seq_len)
        data_dict = generator.generate(
            initial_states=true_initial_states,
            true_params=true_params,
            noise_std=0.01
        )
        x_raw = data_dict["x_raw"]
        y_raw = data_dict["y_raw"]
        self.logger.info(f"Adatok generálva. x_raw forma: {x_raw.shape}, y_raw forma: {y_raw.shape}")

        self.logger.info("=== 2. FÁZIS: ODE Előkalibráció (Kutatói feltevés illesztése) ===")
        calibrator = ODEPreCalibrator(assumed_ode_model=assumed_model, d_p=assumed_d_p, device=self.train_config.device)
        aux_raw = calibrator.calibrate(y_raw=y_raw, epochs=600, lr=0.01)
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
        # A paramétereket nagyjából a valósághoz közeli, de nem tökéletes értékekről indítjuk
        models = EINNBuilder.build_einn(
            model_config=self.model_config,
            train_config=self.train_config,
            param_calibration={"beta": 0.3, "alpha": 0.1, "gamma": 0.1, "mu": 0.01},
            model_type="SEIRM",
            seed=42
        )
        # JAVÍTÁS: Célváltozó indexének biztonságos átadása felülírás nélkül
        models.ode_model.target_idx = assumed_model.target_idx
        self.logger.info("Modellek sikeresen inicializálva.")

        self.logger.info("=== 5. FÁZIS: EINN Hálózat Tanítása ===")
        trainer = EINNTrainer(models=models, config=self.train_config)
        metrics = trainer.train(dataset=dataset, epochs=epochs, batch_size=batch_size)

        self.logger.info("=== 6. FÁZIS: Utófeldolgozás és Mentés ===")
        post_processor = TrainingPostProcessor(output_dir=self.output_dir)
        post_processor.save_models(models=models, prefix="einn_baseline")
        post_processor.save_metrics(metrics=metrics)
        post_processor.plot_loss_curves(metrics=metrics, filename="baseline_loss_curves.png")

        np.save(os.path.join(self.output_dir, "true_latent_states.npy"), data_dict["true_latent_states"])

        self.logger.info(f"Kísérlet befejezve! Eredmények a(z) {self.output_dir} mappában.")
        return models, metrics, preprocessor


if __name__ == "__main__":
    # ---------------------------------------------------------
    # TÖKÉLETES BASELINE BEÁLLÍTÁS (VALÓSÁG == FELTEVÉS)
    # ---------------------------------------------------------

    # Valóság: SEIRM modell, Célváltozó: Mortalitás (index: 4)
    true_physics = SEIRMModel(target_idx=4)

    # Feltevés: Szintén SEIRM, célváltozó ugyanaz
    assumed_physics = SEIRMModel(target_idx=4)

    # Megnövelt kapacitás, hogy a Feature modul biztosan képes legyen a komoly tanulásra
    model_cfg = EINNModelConfig(
        d_x=2,
        d_e=32,  # 32 dimenziós beágyazási tér
        d_s=5,
        d_p=4,
        feature_n_layers=2  # 2 rejtett réteg az Encoder-Decoderben
    )

    # Baseline súlyok: Mivel a feltelezett modell tökéletes, minden fizikai és aux
    # veszteség bátran kaphat 1.0-s súlyt, nincsenek egymásnak ellentmondó gradiensek.
    train_cfg = EINNTrainConfig(
        device='cuda',
        learning_rate=0.005,
        epochs_per_phase=1000,
        early_stopping_patience=30,
        early_stopping_min_delta=1e-5,
        reps={
            'phase_1': 1,
            'phase_2': 1,
            'phase_3': 1,
            'phase_4': 1
        },
        loss_weights={
            'data_T': 1.0, 'data_F': 1.0,
            'ode_T': 1.0, 'ode_F': 1.0,
            'ode_future_T': 1.0, 'ode_future_F': 1.0,
            'kd_target': 1.0, 'kd_emb': 0.5,
            'mono': 1.0, 'param': 0.001,
            'aux': 0.5  # Mivel a PreCalibrator is SEIRM-et használt, ez a mankó most tökéletes!
        }
    )

    runner = EINNRunner(
        output_dir="./experiment_results/baseline_seirm_run",
        model_config=model_cfg,
        train_config=train_cfg
    )

    runner.run_experiment(
        true_model=true_physics,
        true_initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],  # S, E, I, R, M
        true_params=[0.4, 0.2, 0.1, 0.05],  # beta, alpha, gamma, mu
        assumed_model=assumed_physics,
        assumed_d_p=4,
        seq_len=200,  # Elegendő hossz a járvány felfutásához és lecsengéséhez
        window_size=14,  # Két hetes ablakok
        epochs=80,  # Adunk neki 80 makró-epochot a tökéletes finomhangoláshoz
        batch_size=16
    )
