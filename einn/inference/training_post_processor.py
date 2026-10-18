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

    def plot_loss_curves(self, metrics: list[TrainingMetrics], filename: str = "loss_curves.png"):
        """Plots the macro-epoch loss curves based on the collected metrics."""
        if not metrics:
            return

        import matplotlib.pyplot as plt

        # Phase 4 metrikák kigyűjtése az epochok végén
        phase4_metrics = [m for m in metrics if m.phase == 4]
        if not phase4_metrics:
            return

        epochs = [m.epoch for m in phase4_metrics]
        total_loss = [m.total_loss for m in phase4_metrics]

        # Biztonságos kigyűjtés (ha egy loss komponens hiányzik, 0-t teszünk be)
        data_t_loss = [m.loss_components.get('data_T', 0.0) for m in phase4_metrics]
        data_f_loss = [m.loss_components.get('data_F', 0.0) for m in phase4_metrics]
        ode_t_loss = [m.loss_components.get('ode_T', 0.0) for m in phase4_metrics]
        aux_loss = [m.loss_components.get('aux', 0.0) for m in phase4_metrics]
        ic_loss = [m.loss_components.get('ic', 0.0) for m in phase4_metrics]

        plt.figure(figsize=(14, 8))
        plt.plot(epochs, total_loss, label='Total Loss (Phase 4)', color='black', linewidth=2.5)
        plt.plot(epochs, data_t_loss, label='Data Loss (Time)', color='blue', linestyle='--', alpha=0.7)
        plt.plot(epochs, data_f_loss, label='Data Loss (Feature)', color='cyan', linestyle='--', alpha=0.7)
        plt.plot(epochs, ode_t_loss, label='ODE Physics Loss (Time)', color='red', linestyle=':', alpha=0.7)

        # Új görbék hozzáadása
        plt.plot(epochs, aux_loss, label='Auxiliary (Mankó) Loss', color='green', linestyle='-', alpha=0.6)
        plt.plot(epochs, ic_loss, label='IC Loss', color='purple', linestyle='-.', alpha=0.6)

        plt.yscale('log')
        plt.xlabel('Makro-Epoch')
        plt.ylabel('Veszteség (Loss Value) - Log Skála')
        plt.title('EINN Tanítási Veszteséggörbék Kialakulása')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='upper right')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Bővített veszteséggörbe ábra elmentve ide: {filepath}")
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
