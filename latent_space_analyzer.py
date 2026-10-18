import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.model.einn_builder import EINNBuilder
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.data.data_preprocessor import DataPreprocessor
from einn.ode.sir_model import SIRModel


class LatentSpaceAnalyzer:
    """
    Kinyeri és vizualizálja a Feature Module által generált látens beágyazásokat (e_t)
    a kontroll (Sanity) és a teszt (Experiment) kísérletek összehasonlítására.
    """

    def __init__(self, sanity_dir: str, experiment_dir: str, output_dir: str = "./experiment_results/latent_analysis"):
        self.sanity_dir = sanity_dir
        self.experiment_dir = experiment_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Egységes konfiguráció (ugyanaz, mint a runnerben)
        self.model_cfg = EINNModelConfig(d_x=5, d_e=32, d_s=5, d_p=4, feature_n_layers=2)
        self.train_cfg = EINNTrainConfig(device='cpu')  # Csak inference-hez kell

    def _generate_test_data(self):
        """Generálunk egy egységes, friss adatsort, amin mindkét modellt leteszteljük."""
        print("Adatok generálása az elemzéshez...")
        generator = SyntheticDataGenerator(true_ode_model=SIRModel(target_idx=1), seq_len=200)
        data_dict = generator.generate(initial_states=[0.99, 0.01, 0.0], true_params=[0.3, 0.1], noise_std=0.0)

        preprocessor = DataPreprocessor()
        batch_dict = preprocessor.fit_transform(
            x_raw=data_dict["x_raw"],
            y_raw=data_dict["y_raw"],
            aux_raw=data_dict["x_raw"],  # Ideiglenes
            target_idx=1
        )
        return batch_dict["x"], batch_dict["t"]

    def _load_feature_module(self, model_dir: str, model_type: str):
        """Betölti a lementett modellsúlyokat a specifikus feature_module fájlból."""
        init_params = {"beta": 0.3, "alpha": 0.2, "gamma": 0.1, "mu": 0.05}

        models = EINNBuilder.build_einn(
            model_config=self.model_cfg,
            train_config=self.train_cfg,
            param_calibration=init_params,
            model_type=model_type,
            seed=42
        )

        # JAVÍTÁS: Célzottan a feature module súlyait töltjük be
        feature_model_path = os.path.join(model_dir, "einn_experiment_feature_module.pth")
        if not os.path.exists(feature_model_path):
            raise FileNotFoundError(f"Nem található a modell fájl: {feature_model_path}")

        models.feature_module.load_state_dict(torch.load(feature_model_path, map_location=self.train_cfg.device))
        models.feature_module.eval()

        return models.feature_module

    def extract_embeddings(self):
        x, t = self._generate_test_data()

        x_tensor = x.unsqueeze(0).to(self.train_cfg.device)
        t_tensor = t.unsqueeze(0).to(self.train_cfg.device)

        print("Sanity modell betöltése és beágyazások kinyerése...")
        sanity_feature_module = self._load_feature_module(self.sanity_dir, "SEIRM")
        with torch.no_grad():
            sanity_embeddings = sanity_feature_module(x=x_tensor, t=t_tensor, mask=None)

        print("Experiment modell betöltése és beágyazások kinyerése...")
        exp_feature_module = self._load_feature_module(self.experiment_dir, "SEIRM")
        with torch.no_grad():
            exp_embeddings = exp_feature_module(x=x_tensor, t=t_tensor, mask=None)

        return sanity_embeddings.squeeze(0).cpu().numpy(), exp_embeddings.squeeze(0).cpu().numpy()

    def plot_comparative_analysis(self, sanity_emb: np.ndarray, exp_emb: np.ndarray):
        """Lefuttatja a PCA-t és a t-SNE-t, majd kirajzolja őket."""
        print("PCA és t-SNE számítása (ez eltarthat néhány másodpercig)...")

        time_colors = np.arange(sanity_emb.shape[0])

        # PCA
        pca = PCA(n_components=2)
        sanity_pca = pca.fit_transform(sanity_emb)
        exp_pca = pca.fit_transform(exp_emb)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        sanity_tsne = tsne.fit_transform(sanity_emb)
        exp_tsne = tsne.fit_transform(exp_emb)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # PCA Plotok
        sc1 = axes[0, 0].scatter(sanity_pca[:, 0], sanity_pca[:, 1], c=time_colors, cmap='viridis', s=20)
        axes[0, 0].set_title("PCA: Sanity Check (SEIRM -> SEIRM)\nHelyes Struktúra")
        axes[0, 0].grid(True, linestyle='--', alpha=0.5)

        axes[0, 1].scatter(exp_pca[:, 0], exp_pca[:, 1], c=time_colors, cmap='viridis', s=20)
        axes[0, 1].set_title("PCA: Experiment (SIR -> SEIRM)\nStrukturális Hiba (SME)")
        axes[0, 1].grid(True, linestyle='--', alpha=0.5)

        # t-SNE Plotok
        axes[1, 0].scatter(sanity_tsne[:, 0], sanity_tsne[:, 1], c=time_colors, cmap='viridis', s=20)
        axes[1, 0].set_title("t-SNE: Sanity Check (SEIRM -> SEIRM)\nHelyes Struktúra")
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)

        axes[1, 1].scatter(exp_tsne[:, 0], exp_tsne[:, 1], c=time_colors, cmap='viridis', s=20)
        axes[1, 1].set_title("t-SNE: Experiment (SIR -> SEIRM)\nStrukturális Hiba (SME)")
        axes[1, 1].grid(True, linestyle='--', alpha=0.5)

        cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.8)
        cbar.set_label('Időlépés (Napok)')

        plt.suptitle("Látens Tér ($e_t$) Összehasonlító Elemzése", fontsize=18)

        out_path = os.path.join(self.output_dir, "latent_space_comparison.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Az elemzés ábrája sikeresen elmentve: {out_path}")


if __name__ == "__main__":
    SANITY_DIR = "./experiment_results/01_sanity_seirm"
    EXP_DIR = "./experiment_results/02_experiment_sir_seirm"

    analyzer = LatentSpaceAnalyzer(sanity_dir=SANITY_DIR, experiment_dir=EXP_DIR)

    sanity_e, exp_e = analyzer.extract_embeddings()
    analyzer.plot_comparative_analysis(sanity_e, exp_e)