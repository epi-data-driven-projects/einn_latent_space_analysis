
from typing import Any, Dict

import numpy as np
from scipy.spatial import procrustes
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import paired_cosine_distances
import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn.inference.inference_engine import InferenceEngine
from einn.model.interface.einn_models import EINNModels


class EINNEvaluator:
    """
    Independent Evaluation Engine for the EINN Framework.

    This class handles mathematical and statistical evaluation of a trained
    EINN model on unseen test data.
    """

    def __init__(self, models: EINNModels, device: str = 'cpu'):
        """
        Initializes the evaluator with a pre-trained EINN model.

        :param models: The pre-trained neural networks.
        :param device: The target computation device.
        """
        self.device = torch.device(device)
        self.models = models

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
        Evaluates one trajectory using the available historical observations.

        The method extracts the complete time-dependent latent representations.
        PCA is intentionally not performed here, because multiple trajectories
        must be projected using the same fitted PCA transformation.
        """

        results = {}

        x_test_slice = torch.as_tensor(
            x_test[:t_end],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

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

            if t_end < len(true_target_trajectory):
                mse_future = float(
                    np.mean(
                        (
                            true_target_trajectory[t_end:]
                            - predicted_target[t_end:]
                        ) ** 2
                    )
                )
            else:
                mse_future = float(
                    np.mean(
                        (true_target_trajectory - predicted_target) ** 2
                    )
                )

            results["predicted_trajectory"] = predicted_target
            results["mse_future"] = mse_future

            # Scale time
            t_min = scaling_stats.t_min.to(future_t.device)
            t_max = scaling_stats.t_max.to(future_t.device)

            future_t_scaled = (
                (future_t - t_min)
                / (t_max - (t_min + 1e-8))
            )

            # Time Module embedding
            e_t_tensor = self.models.time_module(t=future_t_scaled)
            emb_t = e_t_tensor.squeeze(0).cpu().numpy()

            # Feature Module embedding
            feature_output = self.models.feature_module(
                x=x_test_slice,
                t=future_t_scaled,
                mask=None
            )

            # FeatureModule may return a structured output object.
            # In that case, use its time-dependent embeddings field.
            if hasattr(feature_output, "embeddings"):
                e_f_tensor = feature_output.embeddings
            else:
                e_f_tensor = feature_output

            emb_f = e_f_tensor.squeeze(0).cpu().numpy()

            results["latent_time"] = emb_t
            results["latent_feature"] = emb_f

        # t-SNE remains trajectory-specific for now.
        # Do not directly compare coordinates from separate t-SNE fits.
        tsne = TSNE(
            n_components=2,
            perplexity=10.0,
            random_state=42
        )

        tsne_coords = tsne.fit_transform(emb_f)
        results["tsne_coords"] = tsne_coords

        # Procrustes Analysis
        _, _, disparity = procrustes(emb_t, emb_f)
        results["procrustes_disparity"] = float(disparity)

        # Paired Cosine Similarity
        cos_sim = 1 - paired_cosine_distances(emb_t, emb_f)
        results["cosine_similarity"] = cos_sim

        return results
