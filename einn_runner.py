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
        self.train_config.t_max = preprocessor.t_max

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

        self.logger.info("=== 7. FÁZIS: Predikció és Valóság Összehasonlítása ===")
        from einn.inference.inference_engine import InferenceEngine

        engine = InferenceEngine(models)

        # A Batch dimenzió hozzáadása a teljes idősorhoz [1, Seq_len, Features]
        full_x = batch_dict["x"].unsqueeze(0).to(self.train_config.device)
        full_t = batch_dict["t"].unsqueeze(0).to(self.train_config.device)

        # Predikció lekérése a Feature Modulon keresztül (Az InferenceEngine visszaskálázza a valós térbe)
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

        self.logger.info(f"Kísérlet befejezve! Eredmények a(z) {self.output_dir} mappában.")
        return models, metrics, preprocessor


# ==========================================
# SANITY CHECK: Működik-e egyáltalán a hálózat?
# ==========================================
if __name__ == "__main__":
    # 1. A legkönnyebb adat: SIR modell (3 rekesz, 2 paraméter)
    true_physics = SIRModel(target_idx=1)

    # Nincs strukturális hiba: a hálózat is SIR modellt feltételez!
    assumed_physics = SIRModel(target_idx=1)

    model_cfg = EINNModelConfig(
        d_x=5,
        d_e=32,
        d_s=3,  # SIR miatt 3 rekesz!
        d_p=2,  # SIR miatt 2 paraméter!
        feature_n_layers=2
    )

    train_cfg = EINNTrainConfig(
        device='cpu',
        learning_rate=0.002,  # Visszavettük az ugrálások megszüntetésére
        epochs_per_phase=500,
        early_stopping_patience=40,
        reps={
            'phase_1': 1, 'phase_2': 1, 'phase_3': 1, 'phase_4': 1
        },
        loss_weights={
            # Az adat és az offline kalibráció adja az alapot
            'data_T': 1.0, 'data_F': 1.0,
            'aux': 1.0,

            # A FIZIKA VISSZAKAPCSOLÁSA (Kisebb súllyal)
            # Ez fogja "letörni" az S-t és "felemelni" az R-t az I hatására!
            'ode_T': 0.1, 'ode_F': 0.1,

            # Jövőbeli ODE egyelőre maradjon nagyon pici
            'ode_future_T': 0.01, 'ode_future_F': 0.01,

            # Monotonitás bekapcsolása (S csak csökkenhet, R csak nőhet)
            'mono': 0.5,
            'param': 0.01,

            'kd_target': 1.0, 'kd_emb': 1.0,
        }
    )

    runner = EINNRunner(
        output_dir="./experiment_results/sanity_check_sir",
        model_config=model_cfg,
        train_config=train_cfg
    )

    runner.run_experiment(
        true_model=true_physics,
        true_initial_states=[0.99, 0.01, 0.0],  # S, I, R
        true_params=[0.3, 0.1],
        assumed_model=assumed_physics,
        assumed_d_p=2,
        seq_len=200,
        window_size=200,
        epochs=100,
        batch_size=1
    )
