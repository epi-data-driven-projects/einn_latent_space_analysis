import os
import torch
import numpy as np
import random

# Importok a te kódodból
from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_preprocessor import DataPreprocessor
from einn.data.einn_dataset import EINNDataset
from einn.data.ode_precalibrator import ODEPreCalibrator
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.model.einn_builder import EINNBuilder
from einn.ode.seirm_model import SEIRMModel
from einn.training.einn_trainer import EINNTrainer
from einn.data.data_scaling_stats import DataScalingStats  # ÚJ IMPORT

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_mini_training(true_params, scenario_name, device):
    """
    Legenerál 1 db rövid adatsort a megadott valódi paraméterekkel, és lefuttat 
    egy villám-tanítást, hogy megnézzük, mit tanul meg a hálózat.
    """
    print(f"\n[{scenario_name}] Adatgenerálás és villám-tanítás indul...")
    print(f"[{scenario_name}] Valódi generáló paraméterek (CÉL): {true_params}")

    # 1. Adatgenerálás
    true_model = SEIRMModel(target_idx=2) # 2 = Infected
    generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=100)
    
    data = generator.generate(
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
        true_params=true_params,
        noise_std=0.01
    )

    # 2. Mankó kalibráció (nagyon kevés iterációval a sebesség miatt)
    calibrator = ODEPreCalibrator(assumed_ode_model=SEIRMModel(target_idx=2), d_p=4, device=device)
    aux_raw = calibrator.calibrate(y_raw=data["y_raw"], initial_states=[0.99, 0.01, 0.0, 0.0, 0.0], epochs=20, lr=0.1)
    
    # 3. Preprocessing
    preprocessor = DataPreprocessor()
    batch_dict = preprocessor.fit_transform(
        x_raw=data["x_raw"], y_raw=data["y_raw"], aux_raw=aux_raw, target_idx=2
    )
    preprocessor.t_min = torch.tensor(0.0, dtype=torch.float32)
    preprocessor.t_max = torch.tensor(100.0, dtype=torch.float32)

    # 4. Dataset (teljes ablakos, hogy gyors legyen)
    dataset = EINNDataset(
        x=batch_dict["x"].unsqueeze(0),
        y=batch_dict["y"].unsqueeze(0),
        t=batch_dict["t"].unsqueeze(0),
        aux_targets=batch_dict["aux_targets"].unsqueeze(0),
        window_size=100,
        device=device
    )

    # 5. Konfiguráció és DataScalingStats
    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=5, d_p=4)
    train_cfg = EINNTrainConfig(
        device=device, learning_rate=0.005, 
        reps={'phase_1': 10, 'phase_2': 10, 'phase_3': 0, 'phase_4': 0} # CSAK 1. és 2. fázist futtatunk
    )
    
    ic_normalized = (torch.tensor([0.99, 0.01, 0.0, 0.0, 0.0], device=device) - preprocessor.aux_mean.to(device)) / (preprocessor.aux_std.to(device) + 1e-8)
    
    # Létrehozzuk a tiszta DataScalingStats objektumot
    scaling_stats = DataScalingStats(
        aux_mean=preprocessor.aux_mean.to(device),
        aux_std=preprocessor.aux_std.to(device),
        t_min=preprocessor.t_min.to(device),
        t_max=preprocessor.t_max.to(device),
        initial_states=ic_normalized.cpu().tolist()
    )

    # 6. Modell építése (A SEED RÖGZÍTÉSE ITT KRITIKUS)
    seed_everything(42)
    models = EINNBuilder.build_einn(
        model_config=model_cfg, train_config=train_cfg,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05}, # Mindkettő innen indul!
        model_type="SEIRM", seed=42
    )

    initial_learned = models.ode_model.get_scaled_params().detach().cpu().numpy()
    print(f"[{scenario_name}] Hálózat induló (kezdeti) paraméterei: {initial_learned}")

    # 7. Tanítás (Itt adjuk át az új scaling_stats objektumot)
    trainer = EINNTrainer(models=models, config=train_cfg, scaling_stats=scaling_stats)
    trainer.train(dataset=dataset, epochs=1, batch_size=1)

    final_learned = models.ode_model.get_scaled_params().detach().cpu().numpy()
    print(f"[{scenario_name}] Hálózat MEGTANULT paraméterei: {final_learned}")
    
    return final_learned

def test_parameter_learning():
    print("=== ODE Paraméterek Adatvezérelt Tanulásának Tesztje ===")
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # A Szcenárió: Agresszív járvány (Magas beta, alacsony gamma)
    params_A = [0.8, 0.2, 0.05, 0.05]
    learned_A = run_mini_training(params_A, "SZCENÁRIÓ A (Agresszív)", device)

    # B Szcenárió: Enyhe járvány (Alacsony beta, magas gamma)
    params_B = [0.2, 0.2, 0.2, 0.05]
    learned_B = run_mini_training(params_B, "SZCENÁRIÓ B (Enyhe)", device)

    print("\n================ ÖSSZEGZÉS ================")
    print("Mindkét modell [0.4, 0.2, 0.1, 0.05] tippről indult (ugyanolyan random súlyokkal).")
    print(f"Agresszív adatból tanult béta: {learned_A[0]:.4f} (Elvárt irány: Fel)")
    print(f"Enyhe adatból tanult béta:     {learned_B[0]:.4f} (Elvárt irány: Le)")
    
    if learned_A[0] > learned_B[0]:
        print("\n=> SIKER! A hálózat felismerte a különbséget, és az adatokhoz igazította a fizikai paramétereket!")
    else:
        print("\n=> HIBA! A paraméterek nem az adatoknak megfelelően frissültek.")

if __name__ == "__main__":
    test_parameter_learning()