"""
Forecasting and latent-space separation evaluator for trained multi-model EINN models.

Supports held-out Feature-branch inference conditioned on an assumed model label (thesis
Section 2.4) and computes full-dimensional geometric, neighborhood, and k-NN graph
separation metrics—including the Silhouette score analyzed in Sections 2.5 and 4.2.
"""
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist, directed_hausdorff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.inference.inference_engine import InferenceEngine
from einn_multimodel.model.interface.einn_models import EINNModels


class EINNEvaluator:
    """
    Prediction and latent-space evaluator for a trained multi-model EINN.

    Evaluates physical-domain forecasts from the shared Feature and Output modules on
    unseen trajectories and provides static geometric separation diagnostics across
    model families in the original d_e-dimensional latent space.
    """

    def __init__(self, models: EINNModels, device: str = "cpu") -> None:
        """
        Switch all multi-model EINN submodules to evaluation mode and build the inference engine.

        :param EINNModels models: Container holding trained time_modules, feature_module,
            output_module, and ode_models.
        :param str device: Target PyTorch device string (e.g. cpu or cuda:0).
        :return None: Initializes device, evaluation mode, and InferenceEngine in place.
        """
        self.device: torch.device = torch.device(device)
        self.models: EINNModels = models

        # Disable training-only behavior (e.g. dropout) across all neural and ODE submodules.
        self.models.time_modules.eval()
        self.models.feature_module.eval()
        self.models.output_module.eval()
        self.models.ode_models.eval()

        self.engine: InferenceEngine = InferenceEngine(models)

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
        """
        Run held-out Feature-branch forecasting from the first t_end observations and extract embeddings.

        Performs inference using only FeatureModule and OutputModule conditioned on
        assumed_model_label (thesis Section 2.4), without any auxiliary classifier.

        :param torch.Tensor x_test: Standardized test input sequence of shape (T, d_x) or (B, T, d_x).
        :param torch.Tensor future_t: Unscaled evaluation time grid of shape (T_eval, 1) or (B, T_eval, 1).
        :param DataScalingStats scaling_stats: Training-set Z-score and temporal min-max statistics
            for physical-domain inverse transformation and time scaling.
        :param int target_idx: Compartment index j* in [0, d_s) of the target epidemiological state.
        :param np.ndarray true_target_trajectory: Noise-free reference target values of shape (T,).
        :param int t_end: Number of historical timesteps exposed to FeatureModule.
        :param Optional[int] true_model_label: Ground-truth generating model ID (optional metadata).
        :param Optional[int] assumed_model_label: Assumed model ID supplied to FeatureModule;
            required when the trained model registry contains more than one model family.
        :return Dict[str, Any]: Evaluation dictionary containing:
            mse_future (float MSE on reference[t_end:t_end + T_eval]),
            predicted_trajectory of shape (T_eval,),
            latent_feature E_F of shape (T_eval, d_e),
            assumed_model_label (int), and when T_eval >= 3 / >= 4, optional per-trajectory
            pca_variance, pca_dims_99, pca_coords_3d of shape (T_eval, <=3), and tsne_coords (T_eval, 2).
        :raises ValueError: If x_test does not have 2 or 3 dimensions, or if assumed_model_label
            is omitted for a multi-model EINN.
        """
        # Slice historical prefix [0, t_end) and ensure leading batch dimension: (B, t_end, d_x).
        if x_test.ndim == 2:
            history = x_test[:t_end].unsqueeze(0)
        elif x_test.ndim == 3:
            history = x_test[:, :t_end]
        else:
            raise ValueError("x_test must have shape [T,d_x] or [B,T,d_x].")
        history = history.to(self.device, dtype=torch.float32)

        future_t = future_t.to(self.device, dtype=torch.float32)
        if future_t.ndim == 2:
            future_t = future_t.unsqueeze(0)  # Shape: (1, T_eval, 1)

        if assumed_model_label is None and len(self.models.model_to_id) > 1:
            raise ValueError("Multi-model evaluation requires assumed_model_label.")
        label_id = int(assumed_model_label or 0)
        labels = torch.full(
            (history.size(0),), label_id, dtype=torch.long, device=self.device
        )  # Shape: (B,)

        with torch.no_grad():
            # Predict physical-domain compartment states via Feature -> Output: (B, T_eval, d_s).
            states = self.engine.predict_future(
                historical_x=history,
                historical_x_mask=None,
                future_t=future_t,
                scaling_stats=scaling_stats,
                assumed_model_label=labels,
            )

            # Min-max scale evaluation times using training bounds and extract E_F: (T_eval, d_e).
            t_min = scaling_stats.t_min.to(self.device)
            t_max = scaling_stats.t_max.to(self.device)
            t_scaled = (future_t - t_min) / (t_max - t_min).clamp_min(1e-8)
            embedding = self.models.feature_module(
                x=history, t=t_scaled, mask=None, assumed_model_label=labels
            ).embeddings.squeeze(0).cpu().numpy()

        # Extract target compartment forecast (T_eval,) and compute out-of-window MSE.
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

        # Compute trajectory-local PCA variance spectrum and 3D projection when at least 3 points exist.
        if embedding.shape[0] >= 3:
            pca = PCA().fit(embedding)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            result["pca_variance"] = cumulative
            result["pca_dims_99"] = int(np.searchsorted(cumulative, 0.99) + 1)
            result["pca_coords_3d"] = PCA(
                n_components=min(3, embedding.shape[0], embedding.shape[1])
            ).fit_transform(embedding)

        # Compute trajectory-local 2D t-SNE projection when at least 4 points exist.
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
        """
        Compute full-dimensional geometric, neighborhood, and k-NN graph separation metrics.

        Quantifies class separation in the original d_e-dimensional latent space (not a 2D
        projection), including the Silhouette score S(A, B) reported in thesis Sections 2.5
        and 4.2, inter-centroid and intra/inter-class Euclidean distances, symmetric Hausdorff
        distance, k-NN label purity, and undirected k-NN graph average clustering coefficient.

        :param np.ndarray embeddings: Latent embedding matrix of shape (N, D).
        :param np.ndarray labels: Integer class labels of shape (N,) corresponding to embeddings.
        :param int n_neighbors: Neighborhood size k for k-NN purity and clustering diagnostics.
        :return dict[str, float]: Dictionary of scalar separation metrics (silhouette,
            mean_centroid_distance, mean_intra_class_distance, mean_inter_class_distance,
            inter_intra_distance_ratio, mean_pairwise_hausdorff, knn_label_purity, and
            knn_graph_average_clustering).
        :raises ValueError: If embeddings is not 2D, lengths mismatch, or N < 3.
        """
        embeddings = np.asarray(embeddings, dtype=float)
        labels = np.asarray(labels).reshape(-1)
        if embeddings.ndim != 2 or len(embeddings) != len(labels):
            raise ValueError("Expected embeddings [N, D] and labels [N].")
        if len(embeddings) < 3:
            raise ValueError("At least three samples are required for embedding metrics.")

        unique, counts = np.unique(labels, return_counts=True)
        result: dict[str, float] = {}

        # Silhouette score in [-1, 1] requires >= 2 classes with >= 2 samples each (Section 2.5).
        if len(unique) >= 2 and counts.min() >= 2 and len(embeddings) > len(unique):
            result["silhouette"] = float(silhouette_score(embeddings, labels))

        # Mean Euclidean distance between class centroids.
        centroids = {label: embeddings[labels == label].mean(axis=0) for label in unique}
        inter_centroid = [
            float(np.linalg.norm(centroids[a] - centroids[b]))
            for i, a in enumerate(unique)
            for b in unique[i + 1:]
        ]
        if inter_centroid:
            result["mean_centroid_distance"] = float(np.mean(inter_centroid))

        # Pairwise intra-class distances, inter-class distances, and symmetric Hausdorff distances.
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

        # k-NN label purity: fraction of each point's k nearest neighbors sharing its label.
        k = min(max(1, n_neighbors), len(embeddings) - 1)
        neighbor_model = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
        neighbor_indices = neighbor_model.kneighbors(embeddings, return_distance=False)[:, 1:]
        neighbor_labels = labels[neighbor_indices]
        result["knn_label_purity"] = float(
            np.mean(neighbor_labels == labels[:, None])
        )

        # Symmetrize the directed k-NN graph into an undirected adjacency matrix and compute
        # the average local clustering coefficient: 2 * triangles / (deg * (deg - 1)).
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
