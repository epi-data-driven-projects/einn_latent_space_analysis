from typing import Any, Dict

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import paired_cosine_distances
import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn.inference.inference_engine import InferenceEngine
from einn.model.interface.einn_models import EINNModels


class EINNEvaluator:
    """
    Independent Evaluation Engine for the EINN Framework.

    This class handles pure mathematical and statistical evaluation of a trained EINN model
    on unseen test data.
    """

    def __init__(self, models: EINNModels, device: str = 'cpu'):
        """
        Initializes the evaluator with a trained EINN model.

        :param EINNModels models: The pre-trained neural networks (Time, Feature, Output modules).
        :param str device: The target computation device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.models = models

        # Ensure all models are strictly in evaluation mode (disables Dropout/BatchNorm updates)
        self.models.time_module.eval()
        self.models.feature_module.eval()
        self.models.output_module.eval()

        self.engine = InferenceEngine(self.models)

    def evaluate(
            self,
            x_test: torch.Tensor,
            future_t: torch.Tensor,
            scaling_stats: DataScalingStats,
            target_idx: int,
            true_target_trajectory: np.ndarray,
            t_end: int
    ) -> Dict[str, Any]:
        """
        Runs a comprehensive evaluation on a specific temporal window (from t=0 to t_end).
        It calculates prediction errors, extracts latent space trajectories,
        and computes dimensionality reduction (PCA, t-SNE) and alignment metrics (Procrustes).

        :param torch.Tensor x_test: Unseen test exogenous data. Shape: [1, Seq_len, d_x].
        :param torch.Tensor future_t: Scaled continuous time tensor covering the forecast horizon.
        :param DataScalingStats scaling_stats: The scaling statistics for inverse transformation.
        :param int target_idx: The index of the predicted compartment (e.g., 2 for Infected).
        :param np.ndarray true_target_trajectory: The ground truth curve.
        :param int t_end: The time index representing the "present day" (end of historical knowledge).

        :return Dict[str, Any]: A dictionary containing all computed metrics and trajectories.
        """
        results = {}

        # Slicing the historical input up to the present day (t_end)
        x_test_slice = x_test[:, :t_end, :].to(self.device)
        future_t = future_t.to(self.device)

        with torch.no_grad():
            # Forecasting
            states_pred = self.engine.predict_future(
                historical_x=x_test_slice,
                historical_x_mask=None,
                future_t=future_t,
                scaling_stats=scaling_stats
            )
            pred_np = states_pred.squeeze(0).cpu().numpy()
            predicted_target = pred_np[:, target_idx]

            # Calculate Mean Squared Error (MSE) specifically for the forecasted future
            # If t_end == total_length, we calculate MSE for the whole curve.
            if t_end < len(true_target_trajectory):
                mse_future = float(np.mean((true_target_trajectory[t_end:] - predicted_target[t_end:]) ** 2))
            else:
                mse_future = float(np.mean((true_target_trajectory - predicted_target) ** 2))

            results["predicted_trajectory"] = predicted_target
            results["mse_future"] = mse_future

            # Latent Space Extraction
            e_t_tensor = self.models.time_module(t=future_t)
            emb_t = e_t_tensor.squeeze(0).cpu().numpy()

            e_f_tensor = self.models.feature_module(x=x_test_slice, t=future_t, mask=None)
            emb_f = e_f_tensor.squeeze(0).cpu().numpy()

            results["latent_time"] = emb_t
            results["latent_feature"] = emb_f

        # PCA
        pca = PCA()
        pca.fit(emb_f)
        cum_variance = np.cumsum(pca.explained_variance_ratio_)
        # Find how many dimensions are required to explain 99% of the variance
        n_comp_99 = int(np.argmax(cum_variance >= 0.99) + 1)

        pca_coords_3d = PCA(n_components=3).fit_transform(emb_f)

        results["pca_variance"] = cum_variance
        results["pca_dims_99"] = n_comp_99
        results["pca_coords_3d"] = pca_coords_3d

        # t-SNE
        # Using perplexity=10.0 as requested in the original script
        tsne = TSNE(n_components=2, perplexity=10.0, random_state=42)
        tsne_coords = tsne.fit_transform(emb_f)
        results["tsne_coords"] = tsne_coords

        # Procrustes Analysis: Measures how much geometric distortion exists between E_T and E_F
        _, _, disparity = procrustes(emb_t, emb_f)
        results["procrustes_disparity"] = float(disparity)

        # Paired Cosine Similarity: Measures the angular alignment step-by-step
        cos_sim = 1 - paired_cosine_distances(emb_t, emb_f)
        results["cosine_similarity"] = cos_sim

        return results
