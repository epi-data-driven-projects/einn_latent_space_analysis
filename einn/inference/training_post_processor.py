import json
import os
from dataclasses import asdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from einn.model.interface.einn_models import EINNModels
from einn.model.interface.training_metrics import TrainingMetrics


class TrainingPostProcessor:
    """
    Kezeli a betanított modellek, metrikák és veszteséggörbék elmentését és vizualizációját.
    A futtatás végén minden futtatási peremfeltételt szinkronizál a lemezre.
    """

    def __init__(self, output_dir: str):
        """
        Inicializálja a poszt-processzort és létrehozza a kimeneti mappát, ha az még nem létezik.

        :param str output_dir: A könyvtár elérési útja, ahová az eredmények kerülnek.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_models(self, models: EINNModels, prefix: str = "einn") -> None:
        """
        Szelektíven elmenti az EINN keretrendszer moduljainak súlyait (state_dict).

        :param EINNModels models: A betanított modelleket tartalmazó adatosztály.
        :param str prefix: A fájlnevek előtagja.
        """
        if hasattr(models, 'time_module') and models.time_module is not None:
            torch.save(models.time_module.state_dict(), os.path.join(self.output_dir, f"{prefix}_time_module.pth"))

        if hasattr(models, 'feature_module') and models.feature_module is not None:
            torch.save(models.feature_module.state_dict(),
                       os.path.join(self.output_dir, f"{prefix}_feature_module.pth"))

        if hasattr(models, 'output_module') and models.output_module is not None:
            torch.save(models.output_module.state_dict(), os.path.join(self.output_dir, f"{prefix}_output_module.pth"))

        if hasattr(models, 'ode_model') and models.ode_model is not None:
            torch.save(models.ode_model.state_dict(), os.path.join(self.output_dir, f"{prefix}_ode_model.pth"))

        print(f"Modellsúlyok (state_dict) sikeresen elmentve ide: {self.output_dir}")

    def save_metrics(self, metrics: List[TrainingMetrics], filename: str = "training_metrics.json") -> None:
        """
        A tanítás során összegyűjtött metrikákat (TrainingMetrics dataclass listát) 
        egy strukturált JSON fájlba exportálja.

        :param List[TrainingMetrics] metrics: A tréner által gyűjtött metrikák listája.
        :param str filename: A JSON fájl neve.
        """
        filepath = os.path.join(self.output_dir, filename)

        # Dataclass objektumok átalakítása standard Python szótárakká (dict)
        metrics_dict_list = [asdict(m) for m in metrics]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict_list, f, indent=4)

        print(f"Tanítási metrikák JSON formátumban elmentve ide: {filepath}")

    def plot_loss_curves(self, metrics: List[TrainingMetrics], filename: str = "loss_curves.png") -> None:
        """
        Veszteséggörbék (Loss curves) rajzolása és mentése. 
        Kizárólag a 4. fázis (a makro-epoch végső összevont tanulása) metrikáit használja 
        az áttekinthető logaritmikus ábrázoláshoz.

        :param List[TrainingMetrics] metrics: A tréner által gyűjtött metrikák listája.
        :param str filename: A kimeneti ábra fájlneve.
        """
        if not metrics:
            print("Nincsenek metrikák a kirajzoláshoz.")
            return

        # Szűrés a 4. fázis végi állapotokra (ahol már minden veszteségkomponens aktív)
        phase4_metrics = [m for m in metrics if m.phase == 4]

        # Hibatűrés: ha valamiért nem futott le a 4. fázis, az epochok utolsó rögzített metrikáját vesszük
        if not phase4_metrics:
            epochs = sorted(list(set(m.epoch for m in metrics)))
            phase4_metrics = [[m for m in metrics if m.epoch == ep][-1] for ep in epochs]

        epochs = [m.epoch for m in phase4_metrics]
        total_losses = [m.total_loss for m in phase4_metrics]

        # Specifikus (rész)veszteségek kinyerése, ha léteznek a szótárban
        data_t_losses = [m.loss_components.get('data_T', None) for m in phase4_metrics]
        ode_t_losses = [m.loss_components.get('ode_T', None) for m in phase4_metrics]
        data_f_losses = [m.loss_components.get('data_F', None) for m in phase4_metrics]

        plt.figure(figsize=(12, 7))

        # Összesített veszteség (vastag fekete vonal)
        plt.plot(epochs, total_losses, label='Total Loss (Phase 4)', color='black', linewidth=2.5)

        # Adat illeszkedési veszteségek (Time és Feature modul)
        if all(val is not None for val in data_t_losses):
            plt.plot(epochs, data_t_losses, label='Data Loss (Time)', linestyle='--', color='blue', alpha=0.7)
        if all(val is not None for val in data_f_losses):
            plt.plot(epochs, data_f_losses, label='Data Loss (Feature)', linestyle='--', color='cyan', alpha=0.7)

        # Fizikai (ODE) veszteség
        if all(val is not None for val in ode_t_losses):
            plt.plot(epochs, ode_t_losses, label='ODE Physics Loss (Time)', linestyle=':', color='red', alpha=0.7)

        plt.title('EINN Tanítási Veszteséggörbék Kialakulása')
        plt.xlabel('Makro-Epoch')
        plt.ylabel('Veszteség (Loss Value) - Log Skála')
        plt.yscale('log')  # Kiemelten fontos a log skála a PINN/EINN hálózatok széles értéktartománya miatt
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()

        print(f"Veszteséggörbék grafikonja elmentve ide: {filepath}")

    def plot_true_vs_predicted(self, true_states: np.ndarray, predicted_states: np.ndarray,
                               filename: str = "true_vs_predicted.png") -> None:
        """
        Kirajzolja az igazi (szimulált) rekeszeket és a modell által becsült rekeszeket.
        """
        if not true_states.shape == predicted_states.shape:
            print(f"Hiba a rajzolásnál: eltérő dimenziók! True: {true_states.shape}, Pred: {predicted_states.shape}")
            return

        d_s = true_states.shape[1]
        fig, axes = plt.subplots(nrows=d_s, ncols=1, figsize=(10, 2.5 * d_s), sharex=True)

        # Címkék SIR (3) vagy SEIRM (5) esetére
        labels = ['Susceptible (S)', 'Exposed (E)', 'Infected (I)', 'Recovered (R)', 'Mortality (M)'] if d_s == 5 else [
            'Susceptible (S)', 'Infected (I)', 'Recovered (R)']

        for i in range(d_s):
            ax = axes[i] if d_s > 1 else axes
            # Valóság (Fekete)
            ax.plot(true_states[:, i], label=f'True {labels[i]}', color='black', linewidth=2)
            # Predikció (Piros, szaggatott)
            ax.plot(predicted_states[:, i], label=f'Predicted {labels[i]}', color='red', linestyle='--', linewidth=2)
            ax.set_ylabel('Populáció aránya')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right')

        plt.xlabel('Napok (Időlépések)')
        plt.suptitle('Valós (Ground Truth) vs. Becsült (Predicted) Látens Állapotok', fontsize=14)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Valós vs. Becsült görbék ábrája elmentve ide: {filepath}")
