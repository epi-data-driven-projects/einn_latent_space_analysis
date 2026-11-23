from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist, directed_hausdorff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.inference.inference_engine import InferenceEngine
from einn_multimodel.model.interface.einn_models import EINNModels


class EINNEvaluator:
    """Prediction and latent-space evaluator for a trained multi-model EINN."""

    def __init__(self, models: EINNModels, device: str = "cpu"):
        """Initialize EINNEvaluator to evaluate original-scale forecasts and compare learned latent trajectories.


        :param EINNModels models: The models containing the time, feature, output, and ODE modules.
        :param str device: PyTorch device used for computations.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.device = torch.device(device)
        self.models = models
        self.models.time_modules.eval()
        self.models.feature_module.eval()
        self.models.output_module.eval()
        self.models.ode_models.eval()
        self.engine = InferenceEngine(models)

    def evaluate(
            self,
            x_test: torch.Tensor,
            future_t: torch.Tensor,
            scaling_stats: DataScalingStats,
            target_idx: int,
            true_target_trajectory: np.ndarray,
            t_end: int,
            true_model_label: Optional[int] = None,
            assumed_model_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate held-out Feature inference, without any classifier.

        :param torch.Tensor x_test: Held-out feature observations supplied to the inference engine.
        :param torch.Tensor future_t: Requested evaluation times for reconstruction and forecasting.
        :param DataScalingStats scaling_stats: Statistics fitted on training data for physical-domain conversion.
        :param int target_idx: Index of the epidemiological target compartment.
        :param np.ndarray true_target_trajectory: Reference target values over the complete evaluation horizon.
        :param int t_end: Number of historical time coordinates exposed to the Feature module.
        :param Optional[int] true_model_label: Identifier of the data-generating mechanistic model.
        :param Optional[int] assumed_model_label: Identifier of the model hypothesis supplied to the network.
        :return Dict[str, Any]: Named tensors, metrics, or model components produced by this operation.
        """
        if x_test.ndim == 2:
            history = x_test[:t_end].unsqueeze(0)
        elif x_test.ndim == 3:
            history = x_test[:, :t_end]
        else:
            raise ValueError("x_test must have shape [T,d_x] or [B,T,d_x].")
        history = history.to(self.device, dtype=torch.float32)
        future_t = future_t.to(self.device, dtype=torch.float32)
        if future_t.ndim == 2:
            future_t = future_t.unsqueeze(0)

        if assumed_model_label is None and len(self.models.model_to_id) > 1:
            raise ValueError("Multi-model evaluation requires assumed_model_label.")
        label_id = int(assumed_model_label or 0)
        labels = torch.full(
            (history.size(0),), label_id, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            states = self.engine.predict_future(
                historical_x=history,
                historical_x_mask=None,
                future_t=future_t,
                scaling_stats=scaling_stats,
                assumed_model_label=labels,
            )
            t_min = scaling_stats.t_min.to(self.device)
            t_max = scaling_stats.t_max.to(self.device)
            t_scaled = (future_t - t_min) / (t_max - t_min).clamp_min(1e-8)
            embedding = self.models.feature_module(
                x=history, t=t_scaled, mask=None, assumed_model_label=labels
            ).embeddings.squeeze(0).cpu().numpy()

        prediction = states.squeeze(0).cpu().numpy()[:, target_idx]
        reference = np.asarray(true_target_trajectory).reshape(-1)
        future_true = reference[t_end:t_end + len(prediction)]
        mse = (
            float(np.mean((future_true - prediction[:len(future_true)]) ** 2))
            if future_true.size else float("nan")
        )
        result: Dict[str, Any] = {
            "mse_future": mse,
            "predicted_trajectory": prediction,
            "latent_feature": embedding,
            "assumed_model_label": label_id,
        }
        if embedding.shape[0] >= 3:
            pca = PCA().fit(embedding)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            result["pca_variance"] = cumulative
            result["pca_dims_99"] = int(np.searchsorted(cumulative, 0.99) + 1)
            result["pca_coords_3d"] = PCA(
                n_components=min(3, embedding.shape[0], embedding.shape[1])
            ).fit_transform(embedding)
        if embedding.shape[0] >= 4:
            result["tsne_coords"] = TSNE(
                n_components=2,
                perplexity=min(10.0, float(embedding.shape[0] - 1)),
                random_state=42,
            ).fit_transform(embedding)
        return result

    @staticmethod
    def evaluate_embedding_separation(
            embeddings: np.ndarray,
            labels: np.ndarray,
            n_neighbors: int = 5,
    ) -> dict[str, float]:
        """Compute geometric, neighborhood, and graph separation diagnostics.

        :param np.ndarray embeddings: Time-dependent latent representations produced by the neural module.
        :param np.ndarray labels: Class identifiers assigned to the samples being compared.
        :param int n_neighbors: Number of nearest neighbors used for the geometric diagnostic.
        :return dict[str, float]: Named tensors, metrics, or model components produced by this operation.
        """
        embeddings = np.asarray(embeddings, dtype=float)
        labels = np.asarray(labels).reshape(-1)
        if embeddings.ndim != 2 or len(embeddings) != len(labels):
            raise ValueError("Expected embeddings [N, D] and labels [N].")
        if len(embeddings) < 3:
            raise ValueError("At least three samples are required for embedding metrics.")

        unique, counts = np.unique(labels, return_counts=True)
        result: dict[str, float] = {}
        if len(unique) >= 2 and counts.min() >= 2 and len(embeddings) > len(unique):
            result["silhouette"] = float(silhouette_score(embeddings, labels))

        centroids = {label: embeddings[labels == label].mean(axis=0) for label in unique}
        inter_centroid = [
            float(np.linalg.norm(centroids[a] - centroids[b]))
            for i, a in enumerate(unique)
            for b in unique[i + 1:]
        ]
        if inter_centroid:
            result["mean_centroid_distance"] = float(np.mean(inter_centroid))

        within_distances = []
        between_distances = []
        hausdorff_distances = []
        for i, class_a in enumerate(unique):
            points_a = embeddings[labels == class_a]
            distance_a = cdist(points_a, points_a)
            if len(points_a) > 1:
                within_distances.extend(distance_a[np.triu_indices(len(points_a), k=1)].tolist())
            for class_b in unique[i + 1:]:
                points_b = embeddings[labels == class_b]
                pairwise = cdist(points_a, points_b)
                between_distances.extend(pairwise.ravel().tolist())
                hausdorff_ab = directed_hausdorff(points_a, points_b)[0]
                hausdorff_ba = directed_hausdorff(points_b, points_a)[0]
                hausdorff_distances.append(float(max(hausdorff_ab, hausdorff_ba)))

        mean_within = float(np.mean(within_distances)) if within_distances else 0.0
        mean_between = float(np.mean(between_distances)) if between_distances else 0.0
        result["mean_intra_class_distance"] = mean_within
        result["mean_inter_class_distance"] = mean_between
        result["inter_intra_distance_ratio"] = mean_between / max(mean_within, 1e-12)
        if hausdorff_distances:
            result["mean_pairwise_hausdorff"] = float(np.mean(hausdorff_distances))

        k = min(max(1, n_neighbors), len(embeddings) - 1)
        neighbor_model = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
        neighbor_indices = neighbor_model.kneighbors(embeddings, return_distance=False)[:, 1:]
        neighbor_labels = labels[neighbor_indices]
        result["knn_label_purity"] = float(
            np.mean(neighbor_labels == labels[:, None])
        )

        graph = kneighbors_graph(
            embeddings,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
        )

        graph = graph.maximum(graph.T).tocsr()
        common_neighbor_counts = graph.multiply(graph @ graph)
        triangles = np.asarray(common_neighbor_counts.sum(axis=1)).reshape(-1) / 2.0
        degrees = np.asarray(graph.sum(axis=1)).reshape(-1)
        possible_triangles = degrees * (degrees - 1.0) / 2.0
        local_clustering = np.divide(
            triangles,
            possible_triangles,
            out=np.zeros_like(triangles, dtype=float),
            where=possible_triangles > 0,
        )
        result["knn_graph_average_clustering"] = float(local_clustering.mean())
        return result
