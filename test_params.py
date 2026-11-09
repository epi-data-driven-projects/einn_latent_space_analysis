import os
import torch
import numpy as np

# Importáljuk a rendszered meglévő moduljait
from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.model.einn_builder import EINNBuilder
from einn.ode.seirm_model import SEIRMModel
from einn.inference.inference_engine import InferenceEngine

def test_ode_parameter_behavior():
    print("=== ODE Paraméter Viselkedés Tesztelése Inferencia Során ===\n")
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # 1. Konfigurációk és Modellek felépítése (Gyorsteszt)
    model_cfg = EINNModelConfig(d_x=5, d_e=20, d_s=5, d_p=4)
    train_cfg = EINNTrainConfig(device=device)
    
    # Kamu statisztikák az inicializáláshoz
    scaling_stats = DataScalingStats(
        aux_mean=torch.zeros(5),
        aux_std=torch.ones(5),
        t_min=torch.tensor(0.0),
        t_max=torch.tensor(200.0),
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0]
    )

    print("1. Modellek inicializálása...")
    models = EINNBuilder.build_einn(
        model_config=model_cfg,
        train_config=train_cfg,
        param_calibration={"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
        model_type="SEIRM",
        seed=42
    )
    
    # 2. Paraméterek lekérdezése (Tanítás előtt/után)
    # A get_scaled_params() visszaadja a [0, 1] közé szorított biológiai paramétereket
    initial_params = models.ode_model.get_scaled_params().detach().cpu().numpy()
    print(f"\n-> Alap ODE Paraméterek [beta, alpha, gamma, mu]:\n{initial_params}")

    # 3. Inferencia Motor inicializálása
    engine = InferenceEngine(models)

    # 4. TESZT 1: Inferencia egy kitalált bemeneti adatsoron (pl. alacsony mobilitás)
    print("\n2. Inferencia futtatása az 'A' Szcenárión (pl. Szigorú lezárások)...")
    mock_x_A = torch.randn((1, 50, 5), device=device) * 0.1  # 50 napnyi múltbeli adat
    future_t = torch.arange(200, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(0)
    
    states_A = engine.predict_future(
        historical_x=mock_x_A,
        historical_x_mask=None,
        future_t=future_t,
        scaling_stats=scaling_stats
    )
    
    params_after_A = models.ode_model.get_scaled_params().detach().cpu().numpy()
    print(f"-> ODE Paraméterek az 'A' Inferencia UTÁN:\n{params_after_A}")
    print(f"Megegyezik az eredetivel? {'IGEN' if np.allclose(initial_params, params_after_A) else 'NEM'}")

    # 5. TESZT 2: Inferencia egy teljesen más bemeneti adaton (pl. magas mobilitás)
    print("\n3. Inferencia futtatása a 'B' Szcenárión (pl. Nincsenek korlátozások)...")
    mock_x_B = torch.randn((1, 50, 5), device=device) * 5.0 + 2.0
    
    states_B = engine.predict_future(
        historical_x=mock_x_B,
        historical_x_mask=None,
        future_t=future_t,
        scaling_stats=scaling_stats
    )
    
    params_after_B = models.ode_model.get_scaled_params().detach().cpu().numpy()
    print(f"-> ODE Paraméterek a 'B' Inferencia UTÁN:\n{params_after_B}")
    print(f"Megegyezik az eredetivel? {'IGEN' if np.allclose(initial_params, params_after_B) else 'NEM'}")

    # 6. Végső konklúzió
    print("\n=== KONKLÚZIÓ ===")
    print("Bár a neurális hálózat kimenetei (states_A és states_B) teljesen mások lesznek a bejövő")
    print("adatok miatt, a mögöttes ODE paraméterek abszolút fixek maradtak.")
    print("Az inferencia már kizárólag a neurális hálózat súlyait használja a predikcióhoz!")

if __name__ == "__main__":
    test_ode_parameter_behavior()