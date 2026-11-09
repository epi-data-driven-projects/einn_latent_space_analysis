import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from einn.data.ode_precalibrator import ODEPreCalibrator
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.seirm_model import SEIRMModel


TARGET_IDX = 2

def check_aux_calibration():
    output_dir = "./experiment_results/debugging"
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    print("1. Véletlenszerű SEIRM állam generálása...")
    true_model = SEIRMModel(target_idx=TARGET_IDX)
    
    # Véletlenszerű fizikai paraméterek (mint a generalization szkriptben)
    true_beta = np.random.uniform(0.3, 0.6)
    true_alpha = np.random.uniform(0.1, 0.3)
    true_gamma = np.random.uniform(0.05, 0.15)
    true_mu = np.random.uniform(0.01, 0.08)
    
    print(f"Valós paraméterek: Beta={true_beta:.3f}, Alpha={true_alpha:.3f}, Gamma={true_gamma:.3f}, Mu={true_mu:.3f}")
    
    generator = SyntheticDataGenerator(true_ode_model=true_model, seq_len=200)
    data = generator.generate(
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
        true_params=[true_beta, true_alpha, true_gamma, true_mu],
        noise_std=0.01
    )
    
    true_states = data["true_latent_states"]
    y_raw = data["y_raw"] # A zajos, megfigyelt Mortalitás (M)
    
    print("2. ODE Előkalibráció (Mankó / Aux Data készítése)...")
    calibrator_model = SEIRMModel(target_idx=TARGET_IDX)
    calibrator = ODEPreCalibrator(assumed_ode_model=calibrator_model, d_p=4, device=device)
    
    # Pontosan ugyanazokkal a beállításokkal, ahogy a fő kód is hívja
    aux_data = calibrator.calibrate(
        y_raw=y_raw, 
        initial_states=[0.99, 0.01, 0.0, 0.0, 0.0], 
        epochs=300, 
        lr=0.001
    )
    
    print("3. Ábrázolás...")
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 18), sharex=True)
    labels = ['Susceptible (S)', 'Exposed (E)', 'Infected (I)', 'Recovered (R)', 'Mortality (M)']
    colors = ['blue', 'orange', 'red', 'green', 'purple']

    for i in range(5):
        ax = axes[i]
        # Valós, zajmentes fizikai szimuláció
        ax.plot(true_states[:, i], label=f'Valós {labels[i]}', color='black', linewidth=2.5)
        
        # Az LBFGS által illesztett Aux Data
        ax.plot(aux_data[:, i], label=f'Illesztett Mankó (Aux) {labels[i]}', color=colors[i], linestyle='--', linewidth=2.5)
        
        # Ha a Mortalitás (M) rekeszen vagyunk, tegyük rá a zajos megfigyelést is
        if i == TARGET_IDX:
            ax.scatter(np.arange(200), y_raw, label='Zajos Megfigyelés (y_raw)', color='gray', s=10, alpha=0.5)
            
        ax.set_ylabel('Populáció aránya')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left')
        ax.set_title(f"{labels[i]} Dinamikája")

    plt.xlabel('Napok (Időlépések)')
    plt.suptitle('Az ODEPreCalibrator által generált Aux Data vs. Valóság', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    filepath = os.path.join(output_dir, "aux_calibration_check.png")
    plt.savefig(filepath, dpi=300)
    print(f"\nAz ábra sikeresen elmentve ide: {filepath}")

if __name__ == "__main__":
    check_aux_calibration()
