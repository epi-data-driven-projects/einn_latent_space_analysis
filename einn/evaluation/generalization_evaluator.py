from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import paired_cosine_distances

from einn.data.data_scaling_stats import DataScalingStats
from einn.data.input_loader import GeneratedSeries
from einn.inference.inference_engine import InferenceEngine
from einn.model.interface.einn_models import EINNModels


@dataclass
class PhaseEvaluation:
    """Evaluation results for one historical-prefix length."""

    name: str
    observed_steps: int
    mse: float
    procrustes_disparity: float
    cosine_similarity: np.ndarray
    predicted_states: np.ndarray
    time_embeddings: np.ndarray
    feature_embeddings: np.ndarray
    tsne_coordinates: np.ndarray
    pca_coordinates: np.ndarray
    pca_cumulative_variance: np.ndarray
    n_components_99: int


@dataclass
class GeneralizationEvaluation:
    """All per-prefix results for a single held-out trajectory."""

    target_idx: int
    target_name: str
    true_target: np.ndarray
    true_states: np.ndarray
    phases: list[PhaseEvaluation]
    full_prediction: np.ndarray


class EINNGeneralizationEvaluator:
    """Run held-out prefix inference and calculate latent-space diagnostics.

    The evaluation follows the earlier experiment: for each selected historical
    prefix, the Feature Module receives only that prefix, while the model is
    queried on the complete 0..T-1 time grid. MSE is calculated on the unseen
    suffix; for a full-length prefix it is calculated on the full trajectory.
    """

    TARGET_NAMES = {
        0: "Susceptible",
        1: "Exposed",
        2: "Infected",
        3: "Recovered",
        4: "Mortality",
    }

    def __init__(
        self,
        device: str,
        target_idx: int,
        tsne_perplexity: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.device = torch.device(device)
        self.target_idx = target_idx
        self.tsne_perplexity = tsne_perplexity
        self.random_state = random_state

    @staticmethod
    def _safe_first_crossing(mask: np.ndarray, fallback: int) -> int:
        indices = np.flatnonzero(mask)
        return int(indices[0]) if indices.size else int(fallback)

    def select_prefixes(self, true_states: np.ndarray) -> list[tuple[str, int]]:
        """Choose rising edge, peak, decay and day-150 prefixes as before."""
        if true_states.ndim != 2 or true_states.shape[1] <= 2:
            raise ValueError("The true state matrix must contain an infected (I) column at index 2.")

        seq_len = true_states.shape[0]
        infected = true_states[:, 2]
        peak_day = int(np.argmax(infected))
        peak_value = float(infected[peak_day])
        time = np.arange(seq_len)

        rise_fallback = max(1, min(seq_len - 1, peak_day // 2 if peak_day > 1 else 1))
        rise_day = self._safe_first_crossing(
            (infected >= peak_value * 0.2) & (time < peak_day),
            rise_fallback,
        )

        decay_fallback = min(seq_len - 1, max(peak_day + 1, (peak_day + seq_len - 1) // 2))
        decay_day = self._safe_first_crossing(
            (infected <= peak_value * 0.5) & (time > peak_day),
            decay_fallback,
        )

        # The original experiment used day 150 as its fourth observation point.
        fixed_day = min(150, seq_len - 1)
        raw_phases = [
            (f"1. Felszálló ág ({rise_day}. nap)", rise_day),
            (f"2. Járványcsúcs ({peak_day}. nap)", peak_day),
            (f"3. Lecsengés ({decay_day}. nap)", decay_day),
            (f"4. Járvány vége ({fixed_day}. nap)", fixed_day),
        ]

        # The old script sliced x[:t_end], so the number stored here is the
        # exclusive endpoint. Enforce at least one historical observation.
        return [(name, max(1, min(seq_len, int(day)))) for name, day in raw_phases]

    @staticmethod
    def _time_grid(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_len, dtype=torch.float32, device=device).view(1, seq_len, 1)

    @staticmethod
    def _normalise_time(
        raw_time: torch.Tensor,
        scaling_stats: DataScalingStats,
    ) -> torch.Tensor:
        t_min = scaling_stats.t_min.to(raw_time.device)
        t_max = scaling_stats.t_max.to(raw_time.device)
        # Match the current InferenceEngine implementation exactly.
        return (raw_time - t_min) / (t_max - (t_min + 1e-8))

    def _latent_diagnostics(
        self,
        time_embeddings: np.ndarray,
        feature_embeddings: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        _, _, disparity = procrustes(time_embeddings, feature_embeddings)
        cosine_similarity = 1.0 - paired_cosine_distances(
            time_embeddings,
            feature_embeddings,
        )

        n_components = min(feature_embeddings.shape[0], feature_embeddings.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(feature_embeddings)
        cumulative_variance = np.nan_to_num(
            np.cumsum(pca.explained_variance_ratio_),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        reached = np.flatnonzero(cumulative_variance >= 0.99)
        n_components_99 = int(reached[0] + 1) if reached.size else n_components

        pca_3d = PCA(n_components=min(3, n_components))
        coordinates = pca_3d.fit_transform(feature_embeddings)
        if coordinates.shape[1] < 3:
            coordinates = np.pad(
                coordinates,
                ((0, 0), (0, 3 - coordinates.shape[1])),
            )

        perplexity = min(
            self.tsne_perplexity,
            max(1.0, (feature_embeddings.shape[0] - 1) / 3.0),
        )
        tsne_coordinates = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.random_state,
            init="pca",
            learning_rate="auto",
        ).fit_transform(feature_embeddings)

        return (
            float(disparity),
            cosine_similarity,
            tsne_coordinates,
            coordinates,
            cumulative_variance,
            n_components_99,
        )

    def evaluate(
        self,
        models: EINNModels,
        test_series: GeneratedSeries,
        x_scaler: Any,
        scaling_stats: DataScalingStats,
    ) -> GeneralizationEvaluation:
        """Evaluate one held-out series at the four historical-prefix lengths."""
        x_scaled = x_scaler.transform(test_series.x_raw).astype(np.float32)
        true_states = np.asarray(test_series.true_latent_states, dtype=np.float32)
        true_target = true_states[:, self.target_idx]
        seq_len = x_scaled.shape[0]

        if true_states.shape[0] != seq_len:
            raise ValueError("Held-out X and true state trajectory have different lengths.")
        if self.target_idx >= true_states.shape[1]:
            raise ValueError("target_idx is outside the true state matrix.")

        raw_time = self._time_grid(seq_len, self.device)
        scaled_time = self._normalise_time(raw_time, scaling_stats)
        inference_engine = InferenceEngine(models)

        models.time_module.eval()
        models.feature_module.eval()
        models.output_module.eval()
        models.ode_model.eval()

        with torch.no_grad():
            time_embedding_tensor = models.time_module(t=scaled_time)
            time_embeddings = time_embedding_tensor.squeeze(0).cpu().numpy()

        phases: list[PhaseEvaluation] = []
        for phase_name, observed_steps in self.select_prefixes(true_states):
            historical_x = torch.as_tensor(
                x_scaled[:observed_steps],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            predicted_tensor = inference_engine.predict_future(
                historical_x=historical_x,
                historical_x_mask=None,
                future_t=raw_time,
                scaling_stats=scaling_stats,
            )
            predicted = predicted_tensor.squeeze(0).cpu().numpy()

            with torch.no_grad():
                feature_embedding = models.feature_module(
                    x=historical_x,
                    t=scaled_time,
                    mask=None,
                ).squeeze(0).cpu().numpy()

            if observed_steps < seq_len:
                mse = mean_squared_error(
                    true_target[observed_steps:],
                    predicted[observed_steps:, self.target_idx],
                )
            else:
                mse = mean_squared_error(
                    true_target,
                    predicted[:, self.target_idx],
                )

            (
                disparity,
                cosine_similarity,
                tsne_coordinates,
                pca_coordinates,
                cumulative_variance,
                n_components_99,
            ) = self._latent_diagnostics(time_embeddings, feature_embedding)

            phases.append(
                PhaseEvaluation(
                    name=phase_name,
                    observed_steps=observed_steps,
                    mse=float(mse),
                    procrustes_disparity=disparity,
                    cosine_similarity=cosine_similarity,
                    predicted_states=predicted,
                    time_embeddings=time_embeddings.copy(),
                    feature_embeddings=feature_embedding,
                    tsne_coordinates=tsne_coordinates,
                    pca_coordinates=pca_coordinates,
                    pca_cumulative_variance=cumulative_variance,
                    n_components_99=n_components_99,
                )
            )

        # The last row is the day-150/full-prefix diagnostic; retain a separate
        # full-history prediction for the true-vs-predicted state plot.
        full_history = torch.as_tensor(
            x_scaled,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        full_prediction = inference_engine.predict_future(
            historical_x=full_history,
            historical_x_mask=None,
            future_t=raw_time,
            scaling_stats=scaling_stats,
        ).squeeze(0).cpu().numpy()

        return GeneralizationEvaluation(
            target_idx=self.target_idx,
            target_name=self.TARGET_NAMES.get(self.target_idx, f"State {self.target_idx}"),
            true_target=true_target,
            true_states=true_states,
            phases=phases,
            full_prediction=full_prediction,
        )
