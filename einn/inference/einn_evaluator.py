"""
Evaluation engine for single-model EINN forecasting and latent-space diagnostics.
"""
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
    Evaluation engine for a trained single-model EINN on held-out test trajectories.

    Runs Feature-branch forecasting over a specified observation prefix and extracts
    time-dependent Time and Feature latent trajectories (E_T, E_F) along with the
    geometric alignment diagnostics defined in thesis Sections 2.5 and 4.1 (Procrustes
    disparity, pointwise cosine similarity, and optional per-trajectory t-SNE).
    """

    def __init__(self, models: EINNModels, device: str = 'cpu') -> None:
        """
        Switch all EINN submodules to evaluation mode and initialize the inference engine.

        :param EINNModels models: Container holding trained time_module, feature_module,
            and output_module networks.
        :param str device: Target PyTorch device string (e.g. 'cpu' or 'cuda:1').
        """
        self.device = torch.device(device)
        self.models = models

        # Disable training-only layers (e.g. dropout) across all neural submodules.
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
        Forecast one test trajectory from its first t_end observations and extract latent diagnostics.

        PCA is intentionally omitted here because multi-window or multi-trajectory comparisons
        require fitting a single shared PCA basis across all trajectories (thesis Section 2.5).

        :param torch.Tensor x_test: Standardized test input sequence of shape (T, d_x)
            (or array-like convertible to a float32 tensor).
        :param torch.Tensor future_t: Unscaled evaluation time grid of shape (1, T_eval, 1).
        :param DataScalingStats scaling_stats: Training-set Z-score and temporal min-max statistics
            used to invert predictions to the physical domain and scale time inputs.
        :param int target_idx: Compartment index j* in [0, d_s) of the target epidemiological state.
        :param np.ndarray true_target_trajectory: Noise-free reference target values of shape (T_eval,).
        :param int t_end: Observation window length; only x_test[:t_end] is exposed to FeatureModule.
        :return Dict[str, Any]: Evaluation dictionary containing:
            'predicted_trajectory' of shape (T_eval,),
            'mse_future' (float MSE on t >= t_end, or full horizon if t_end >= T_eval),
            'latent_time' E_T of shape (T_eval, d_e),
            'latent_feature' E_F of shape (T_eval, d_e),
            'tsne_coords' of shape (T_eval, 2),
            'procrustes_disparity' (float D_Proc), and
            'cosine_similarity' of shape (T_eval,).
        """
        results = {}

        # Expose only the historical prefix [0, t_end) and add batch dimension: (1, t_end, d_x).
        x_test_slice = torch.as_tensor(
            x_test[:t_end],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        future_t = future_t.to(self.device)

        with torch.no_grad():
            # Generate full-horizon physical-domain state forecasts via Feature -> Output: (1, T_eval, d_s).
            states_pred = self.engine.predict_future(
                historical_x=x_test_slice,
                historical_x_mask=None,
                future_t=future_t,
                scaling_stats=scaling_stats
            )

            # Remove batch axis -> (T_eval, d_s) and select the target compartment -> (T_eval,).
            pred_np = states_pred.squeeze(0).cpu().numpy()
            predicted_target = pred_np[:, target_idx]

            # Compute out-of-window forecast MSE strictly on t >= t_end (thesis Section 4.1).
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

            # Min-max scale evaluation times using training bounds: (1, T_eval, 1) -> [0, 1].
            t_min = scaling_stats.t_min.to(future_t.device)
            t_max = scaling_stats.t_max.to(future_t.device)

            future_t_scaled = (
                (future_t - t_min)
                / (t_max - (t_min + 1e-8))
            )

            # Extract TimeModule latent trajectory E_T: (1, T_eval, d_e) -> (T_eval, d_e).
            e_t_tensor = self.models.time_module(t=future_t_scaled)
            emb_t = e_t_tensor.squeeze(0).cpu().numpy()

            # Extract FeatureModule latent trajectory E_F conditioned on x_test[:t_end].
            feature_output = self.models.feature_module(
                x=x_test_slice,
                t=future_t_scaled,
                mask=None
            )

            # Unwrap structured FeatureModule outputs if present to obtain (1, T_eval, d_e).
            if hasattr(feature_output, "embeddings"):
                e_f_tensor = feature_output.embeddings
            else:
                e_f_tensor = feature_output

            emb_f = e_f_tensor.squeeze(0).cpu().numpy()

            results["latent_time"] = emb_t
            results["latent_feature"] = emb_f

        # Fit a trajectory-local 2D t-SNE projection (T_eval, 2); coordinates are not comparable across runs.
        tsne = TSNE(
            n_components=2,
            perplexity=10.0,
            random_state=42
        )

        tsne_coords = tsne.fit_transform(emb_f)
        results["tsne_coords"] = tsne_coords

        # Full-dimensional Procrustes disparity D_Proc between E_T and E_F (thesis Section 2.5).
        _, _, disparity = procrustes(emb_t, emb_f)
        results["procrustes_disparity"] = float(disparity)

        # Pointwise cosine similarity per timestep t: 1 - cosine_distance, shape (T_eval,).
        cos_sim = 1 - paired_cosine_distances(emb_t, emb_f)
        results["cosine_similarity"] = cos_sim

        return results
