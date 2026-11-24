"""
Held-out Feature-branch evaluation and report generation for multi-model EINN experiments.

Collects full-horizon timestep-level Feature embeddings across held-out trajectories,
computes full-dimensional Hausdorff/separation diagnostics, runs multi-window prefix
generalization evaluations (shared 2D PCA, cumulative explained variance, Time-Feature
cosine similarity, Procrustes disparity, and future MSE), and persists CSV, NPZ, and
figure artifacts (thesis Sections 2.4, 2.5, and 4.2).
"""
from __future__ import annotations

import csv
import importlib
import os
from typing import Any

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn_multimodel.inference.inference_engine import InferenceEngine
from einn_multimodel.model.interface.einn_models import EINNModels


class FeatureOnlyEvaluator:
    """
    Evaluation orchestrator for multi-model EINN held-out trajectories and latent-space reports.

    Validates consistency between the training-fitted target scaler and auxiliary compartment
    statistics, extracts timestep-level Feature embeddings without temporal pooling (Section 3),
    and delegates figure rendering to MultiModelPlotter.
    """

    def __init__(
        self,
        models: EINNModels,
        dataset: Any,
        metrics: Any,
        scaling_stats: DataScalingStats,
        device: str | torch.device,
        output_dir: str | os.PathLike[str],
        target_scaler: Any,
    ) -> None:
        """
        Initialize the evaluator, verify target-scaler alignment, and create the output directory.

        :param EINNModels models: Trained multi-model EINN container (time_modules, feature_module,
            output_module, ode_models, and model_to_id registry).
        :param Any dataset: Held-out ExpandingWindowLoader exposing x_list, y_list, t_list,
            and true_model_labels.
        :param Any metrics: Training or experiment metrics object retained for reporting.
        :param DataScalingStats scaling_stats: Training-fitted compartment mean/std and temporal bounds.
        :param str | torch.device device: Target PyTorch device for inference tensors.
        :param str | os.PathLike[str] output_dir: Directory where CSV, NPZ, and figure reports are saved.
        :param Any target_scaler: Training-fitted scikit-learn StandardScaler for the target compartment.
        :return None: Initializes evaluator attributes and creates output_dir in place.
        :raises ValueError: If target_scaler is missing inverse_transform, ODE models disagree
            on target_idx, or aux_mean/aux_std disagrees with target_scaler at target_idx.
        """
        self.experiment = importlib.import_module("experiments.multi_model_experiment")
        self.models: EINNModels = models
        self.dataset: Any = dataset
        self.metrics: Any = metrics
        self.scaling_stats: DataScalingStats = scaling_stats
        self.target_scaler: Any = target_scaler
        if target_scaler is None or not hasattr(target_scaler, "inverse_transform"):
            raise ValueError("Pass the training-fitted y_scaler.")
        self.device: str | torch.device = device
        self.output_dir: str | os.PathLike[str] = output_dir

        # Build inverse label registry (int -> model_name) and verify a single shared target_idx.
        self.id_to_model: dict[int, str] = {v: k for k, v in models.model_to_id.items()}
        self.class_ids: list[int] = sorted(self.id_to_model)
        target_indices = {model.target_idx for model in models.ode_models.values()}
        if len(target_indices) != 1:
            raise ValueError("All models must agree on the target compartment index.")
        self.target_idx: int = next(iter(target_indices))

        # Ensure physical-domain inverse scaling matches between aux_scaler and y_scaler.
        if not np.allclose(
            [float(scaling_stats.aux_mean[self.target_idx]),
             float(scaling_stats.aux_std[self.target_idx])],
            [float(target_scaler.mean_[0]), float(target_scaler.scale_[0])],
            rtol=1e-3, atol=1e-6,
        ):
            raise ValueError("Incompatible train-fitted AUX and target scaling.")
        os.makedirs(output_dir, exist_ok=True)

    def scale_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalize physical time coordinates into [0, 1] using training bounds.

        :param torch.Tensor t: Unscaled time tensor of shape (B, T, 1) or (T, 1).
        :return torch.Tensor: Normalized time tensor of identical shape.
        """
        minimum = self.scaling_stats.t_min.to(self.device)
        maximum = self.scaling_stats.t_max.to(self.device)
        return (t - minimum) / (maximum - minimum).clamp_min(1e-8)

    def clean_target_physical(self, index: int) -> np.ndarray:
        """
        Inverse-transform the standardized ground-truth target trajectory to physical units.

        :param int index: Zero-based trajectory index in self.dataset.y_list.
        :return np.ndarray: Noise-free physical target trajectory of shape (T,) (float64).
        """
        scaled = np.asarray(self.dataset.y_list[index], dtype=np.float64).reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled).reshape(-1)

    def get_time_module_for_label(self, true_id: int) -> torch.nn.Module:
        """
        Retrieve the model-specific TimeModule corresponding to a true model label ID.

        :param int true_id: Integer model identifier present in self.id_to_model.
        :return torch.nn.Module: The TimeModule instance trained for that epidemiological model.
        :raises AttributeError: If self.models exposes neither time_modules nor time_module_dict.
        """
        model_name = self.id_to_model[int(true_id)]
        if hasattr(self.models, "time_modules"):
            return self.models.time_modules[model_name]
        if hasattr(self.models, "time_module_dict"):
            return self.models.time_module_dict[model_name]
        raise AttributeError(
            "Could not find time module dictionary on models. "
            "Adjust get_time_module_for_label()."
        )

    def collect_full_trajectories(self, all_trajectories: bool = False) -> dict[str, np.ndarray]:
        """
        Extract full-horizon, timestep-level Feature embeddings across held-out trajectories.

        Feeds each complete trajectory into FeatureModule without temporal pooling (Section 3)
        and concatenates all timestep embeddings with their true model labels, trajectory IDs,
        and time indices for joint PCA and separation analysis (Section 4.2).

        :param bool all_trajectories: If True, evaluates every trajectory in self.dataset.x_list;
            if False, selects only the first held-out trajectory per model class.
        :return dict[str, np.ndarray]: Dictionary with keys:
            feature_embeddings of shape (N_total, d_e),
            true_labels of shape (N_total,),
            trajectory_ids of shape (N_total,),
            time_indices of shape (N_total,), and
            selected_dataset_indices of shape (N_traj,).
        :raises ValueError: If self.dataset lacks x_list, t_list, or true_model_labels.
        :raises RuntimeError: If all_trajectories is False and any class has no held-out trajectory.
        """
        required = ("x_list", "t_list", "true_model_labels")
        if any(not hasattr(self.dataset, field) for field in required):
            raise ValueError("Test dataset has no full-trajectory attributes.")
        if all_trajectories:
            selected = list(range(len(self.dataset.x_list)))
        else:
            selected = []
            for true_id in self.class_ids:
                matches = [i for i, label in enumerate(self.dataset.true_model_labels)
                           if int(label) == true_id]
                if not matches:
                    raise RuntimeError(f"Missing held-out trajectory for {true_id}.")
                selected.append(matches[0])

        embedding_parts, true_parts, trajectory_parts, time_parts = [], [], [], []
        self.models.feature_module.eval()
        with torch.no_grad():
            for trajectory_id, idx in enumerate(selected):
                # Add batch dimension: x -> (1, T, d_x), t -> (1, T, 1), labels -> (1,).
                x = torch.as_tensor(self.dataset.x_list[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                t = torch.as_tensor(self.dataset.t_list[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                true_id = int(self.dataset.true_model_labels[idx])
                labels = torch.tensor([true_id], dtype=torch.long, device=self.device)

                # Extract unpooled timestep embeddings E_F: (1, T, d_e) -> (T, d_e).
                e_f = self.models.feature_module(
                    x=x, t=self.scale_time(t), mask=None, true_model_label=labels,
                ).embeddings.squeeze(0).cpu().numpy()
                n_steps = len(e_f)
                embedding_parts.append(e_f)
                true_parts.extend([true_id] * n_steps)
                trajectory_parts.extend([trajectory_id] * n_steps)
                time_parts.extend(range(n_steps))

        return {
            "feature_embeddings": np.concatenate(embedding_parts),
            "true_labels": np.asarray(true_parts, dtype=int),
            "trajectory_ids": np.asarray(trajectory_parts, dtype=int),
            "time_indices": np.asarray(time_parts, dtype=int),
            "selected_dataset_indices": np.asarray(selected, dtype=int),
        }

    def evaluate_hausdorff_test_embeddings(self) -> dict[str, float]:
        """
        Compute raw and L2-unit-normalized Hausdorff distance statistics between the two model classes.

        Evaluates distances in the original d_e-dimensional Feature embedding space across all
        held-out trajectories.

        :return dict[str, float]: Dictionary merging raw hausdorff_statistics with
            hausdorff_unit_normalized and hausdorff_p95_unit_normalized.
        """
        data = self.collect_full_trajectories(all_trajectories=True)
        E = data["feature_embeddings"]  # Shape: (N_total, d_e)
        a_id, b_id = (self.models.model_to_id[name] for name in self.experiment.MODEL_NAMES)
        true = data["true_labels"]
        a = E[true == a_id]
        b = E[true == b_id]
        stats = self.experiment.hausdorff_statistics(a, b)

        # Compute scale-invariant Hausdorff statistics on L2-normalized embedding vectors.
        a_unit = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        b_unit = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        normalized_stats = self.experiment.hausdorff_statistics(a_unit, b_unit)
        return {
            **stats,
            "hausdorff_unit_normalized": normalized_stats["hausdorff"],
            "hausdorff_p95_unit_normalized": normalized_stats["hausdorff_p95"],
        }

    def evaluate_prefix_on_trajectory(self, dataset_idx: int, t_end: int) -> dict[str, Any]:
        """
        Forecast a full trajectory from its first t_end observations and compute latent diagnostics.

        Runs Feature-branch inference conditioned on the trajectory's true model label, extracts
        both Feature embeddings E_F and active Time embeddings E_T of shape (T, d_e), and
        computes pointwise cosine similarity, Procrustes disparity D_Proc, and out-of-window MSE.

        :param int dataset_idx: Zero-based trajectory index in self.dataset.
        :param int t_end: Number of historical timesteps exposed to FeatureModule.
        :return dict[str, Any]: Evaluation dictionary containing t_end, predicted_trajectory (T,),
            true_target (T,), latent_time (T, d_e), latent_feature (T, d_e),
            cosine_similarity (T,), procrustes_disparity (float), true_model_id (int),
            true_model_name (str), dataset_idx (int), and mse_future (float on t >= t_end).
        :raises ValueError: If E_T and E_F shapes mismatch or differ from (T, EMBEDDING_DIM).
        """
        self.models.feature_module.eval()
        self.models.output_module.eval()
        self.models.time_modules.eval()
        if hasattr(self.models, "ode_models"):
            for model in self.models.ode_models.values():
                model.eval()

        x_full = np.asarray(self.dataset.x_list[dataset_idx], dtype=np.float32)
        t_full = np.asarray(self.dataset.t_list[dataset_idx], dtype=np.float32)
        true_id = int(self.dataset.true_model_labels[dataset_idx])
        labels = torch.tensor([true_id], dtype=torch.long, device=self.device)  # Shape: (1,)

        # Slice observed prefix [0, t_end) -> (1, t_end, d_x) and full time grid -> (1, T, 1).
        x_prefix = torch.as_tensor(x_full[:t_end], dtype=torch.float32, device=self.device).unsqueeze(0)
        full_t = torch.as_tensor(t_full, dtype=torch.float32, device=self.device).unsqueeze(0)
        scaled_t = self.scale_time(full_t)

        with torch.no_grad():
            # Predict physical-domain states (1, T, d_s) and extract target compartment (T,).
            states = InferenceEngine(self.models).predict_future(
                historical_x=x_prefix,
                historical_x_mask=None,
                future_t=full_t,
                scaling_stats=self.scaling_stats,
                true_model_label=labels,
            )
            pred = states[0, :, self.target_idx].detach().cpu().numpy()

            # Extract Feature embeddings E_F (T, d_e) and active Time embeddings E_T (T, d_e).
            output_feature = self.models.feature_module(
                x=x_prefix, t=scaled_t, mask=None, true_model_label=labels,
            )
            e_feature = self.experiment.extract_embeddings(output_feature)[0].detach().cpu().numpy()

            time_module = self.get_time_module_for_label(true_id)
            e_time = time_module(t=scaled_t)[0].detach().cpu().numpy()

        clean = self.clean_target_physical(dataset_idx)
        if e_time.shape != e_feature.shape or e_feature.shape != (len(clean), self.experiment.EMBEDDING_DIM):
            raise ValueError(
                f"Incompatible time/feature embedding shapes: {e_time.shape}, {e_feature.shape}"
            )
        cosine = self.experiment.paired_cosine_similarity(e_time, e_feature)

        # Guard against degenerate zero-variance trajectories before Procrustes alignment.
        if np.linalg.norm(e_time - e_time.mean(axis=0)) <= 1e-12 or \
                np.linalg.norm(e_feature - e_feature.mean(axis=0)) <= 1e-12:
            procrustes_disparity = float("nan")
        else:
            _, _, procrustes_disparity = procrustes(e_time, e_feature)

        mse_future = float(np.mean((pred[t_end:] - clean[t_end:]) ** 2))

        return {
            "t_end": int(t_end),
            "predicted_trajectory": pred,
            "true_target": clean,
            "latent_time": e_time,
            "latent_feature": e_feature,
            "cosine_similarity": cosine,
            "procrustes_disparity": float(procrustes_disparity),
            "true_model_id": true_id,
            "true_model_name": self.id_to_model[true_id],
            "dataset_idx": int(dataset_idx),
            "mse_future": mse_future,
        }

    def fit_shared_pca_for_phase_results(self, phase_results: list[dict[str, Any]]) -> PCA:
        """
        Fit one shared 2D PCA basis across all observation windows and populate per-window PCA stats.

        Projects each window's latent_feature (T, d_e) onto the shared 2D basis (pca_coords of
        shape (T, 2)) and fits an independent full PCA per window to record cum_variance,
        pca_95, and pca_99 in place (thesis Section 2.5 and 4.1).

        :param list[dict[str, Any]] phase_results: Per-window evaluation dicts modified in place.
        :return PCA: Fitted 2-component PCA estimator shared across all windows.
        """
        all_features = np.concatenate([result["latent_feature"] for result in phase_results], axis=0)
        shared = PCA(n_components=2)
        shared.fit(all_features)
        for result in phase_results:
            features = result["latent_feature"]
            result["pca_coords"] = shared.transform(features)  # Shape: (T, 2)
            phase_pca = PCA(n_components=min(features.shape))
            phase_pca.fit(features)
            cumulative = np.cumsum(phase_pca.explained_variance_ratio_)
            result["cum_variance"] = cumulative
            result["pca_95"] = int(np.searchsorted(cumulative, 0.95) + 1)
            result["pca_99"] = int(np.searchsorted(cumulative, 0.99) + 1)
        return shared

    def plot_multimodel_generalization_4x4(self, *args: Any, **kwargs: Any) -> Any:
        """
        Delegate 4x4 multi-model generalization plotting to MultiModelPlotter.

        :param Any args: Positional arguments forwarded to MultiModelPlotter.plot_multimodel_generalization_4x4.
        :param Any kwargs: Keyword arguments forwarded to MultiModelPlotter.plot_multimodel_generalization_4x4.
        :return Any: Return value of MultiModelPlotter.plot_multimodel_generalization_4x4.
        """
        from einn_multimodel.visualization.multi_model_plotter import MultiModelPlotter
        return getattr(MultiModelPlotter(self), "plot_multimodel_generalization_4x4")(*args, **kwargs)

        _save_plot(fig, output_path)

    def plot_joint_latent_space_hausdorff(self, *args: Any, **kwargs: Any) -> Any:
        """
        Delegate joint multi-model latent-space plotting to MultiModelPlotter.

        :param Any args: Positional arguments forwarded to MultiModelPlotter.plot_joint_latent_space_hausdorff.
        :param Any kwargs: Keyword arguments forwarded to MultiModelPlotter.plot_joint_latent_space_hausdorff.
        :return Any: Return value of MultiModelPlotter.plot_joint_latent_space_hausdorff.
        """
        from einn_multimodel.visualization.multi_model_plotter import MultiModelPlotter
        return getattr(MultiModelPlotter(self), "plot_joint_latent_space_hausdorff")(*args, **kwargs)

        return destination

    def generate_multimodel_window_reports(self) -> None:
        """
        Generate joint held-out latent-space reports and per-model 4x4 generalization reports.

        First evaluates all held-out trajectories jointly to export Hausdorff metrics (CSV),
        the joint 2D PCA trajectory figure (thesis Figures 16-19), and compressed embeddings
        (all_heldout_feature_embeddings.npz). Then selects one representative held-out trajectory
        per model class and writes its 4x4 figure, NPZ archive, and per-window metrics CSV.

        :return None: Writes all CSV, NPZ, and figure artifacts into self.output_dir.
        :raises RuntimeError: If any model class has no held-out trajectory in self.dataset.
        """
        # Collect full-horizon, unpooled Feature embeddings across all held-out trajectories.
        all_test_data = self.collect_full_trajectories(all_trajectories=True)
        E = all_test_data["feature_embeddings"]  # Shape: (N_total, d_e)
        truth = all_test_data["true_labels"]
        a_id, b_id = (self.models.model_to_id[name] for name in self.experiment.MODEL_NAMES)

        # Compute raw and L2-normalized Hausdorff distances between the two model classes.
        stats = self.experiment.hausdorff_statistics(E[truth == a_id], E[truth == b_id])
        a, b = E[truth == a_id], E[truth == b_id]
        ua = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        ub = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        unit_stats = self.experiment.hausdorff_statistics(ua, ub)
        hausdorff_metrics = {
            **stats,
            "hausdorff_unit_normalized": unit_stats["hausdorff"],
            "hausdorff_p95_unit_normalized": unit_stats["hausdorff_p95"],
        }

        # Persist Hausdorff summary CSV, joint latent-space figure, and raw held-out embeddings.
        with open(os.path.join(self.output_dir, "hausdorff_test_embeddings.csv"),
                  "w", newline="", encoding="utf-8") as handle:
            info = {
                "model_a": self.experiment.MODEL_NAMES[0], "model_b": self.experiment.MODEL_NAMES[1],
                "num_points_a": len(a), "num_points_b": len(b),
                "embedding_dim": E.shape[1], **hausdorff_metrics,
            }
            writer = csv.DictWriter(handle, fieldnames=list(info))
            writer.writeheader()
            writer.writerow(info)
        self.plot_joint_latent_space_hausdorff(all_test_data, hausdorff_metrics)
        np.savez_compressed(
            os.path.join(self.output_dir, "all_heldout_feature_embeddings.npz"),
            **all_test_data,
        )

        # Select the first held-out trajectory for each model class for 4x4 window analysis.
        selected = []
        for true_id in self.class_ids:
            indices = [i for i, tr in enumerate(self.dataset.true_model_labels)
                       if int(tr) == true_id]
            if not indices:
                raise RuntimeError(f"No held-out trajectory for model id {true_id}.")
            selected.append(indices[0])

        # Evaluate all configured observation windows per representative trajectory and save artifacts.
        for dataset_idx in selected:
            phase_results = []
            for t_end in self.experiment.OBSERVATION_WINDOWS:
                phase_results.append(self.evaluate_prefix_on_trajectory(
                    dataset_idx=dataset_idx, t_end=t_end,
                ))
            shared_pca = self.fit_shared_pca_for_phase_results(phase_results)
            model_name = phase_results[0]["true_model_name"]
            prefix = (f"multimodel_generalization_{'_'.join(m.lower() for m in self.experiment.MODEL_NAMES)}_"
                      f"true_{model_name.lower()}_de_{self.experiment.EMBEDDING_DIM}_4x4")
            output_path = os.path.join(self.output_dir, prefix + ".png")

            self.plot_multimodel_generalization_4x4(
                phase_results=phase_results,
                shared_pca=shared_pca,
                output_path=output_path,
                hausdorff_metrics=hausdorff_metrics,
            )
            np.savez_compressed(
                os.path.join(self.output_dir, prefix + ".npz"),
                observation_windows=np.asarray(self.experiment.OBSERVATION_WINDOWS),
                true_target=phase_results[0]["true_target"],
                predictions=np.stack([r["predicted_trajectory"] for r in phase_results]),
                time_embeddings=np.stack([r["latent_time"] for r in phase_results]),
                feature_embeddings=np.stack([r["latent_feature"] for r in phase_results]),
                cosine_similarities=np.stack([r["cosine_similarity"] for r in phase_results]),
                pca_coords=np.stack([r["pca_coords"] for r in phase_results]),
                shared_pca_variance_ratio=shared_pca.explained_variance_ratio_,
                cumulative_pca_variance=np.stack([r["cum_variance"] for r in phase_results]),
                pca_d95=np.asarray([r["pca_95"] for r in phase_results]),
                pca_d99=np.asarray([r["pca_99"] for r in phase_results]),
                procrustes=np.asarray([r["procrustes_disparity"] for r in phase_results]),
                future_mse=np.asarray([r["mse_future"] for r in phase_results]),
            )
            with open(os.path.join(self.output_dir, prefix + "_metrics.csv"),
                      "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["observed_steps", "future_mse", "pca_d95", "pca_d99",
                                 "procrustes_disparity", "mean_cosine"])
                for r in phase_results:
                    writer.writerow([r["t_end"], r["mse_future"], r["pca_95"],
                                     r["pca_99"], r["procrustes_disparity"],
                                     float(np.nanmean(r["cosine_similarity"]))])
            print("4x4 generalization figure saved:", output_path)

    def generate_all_reports(self) -> None:
        """
        Run the complete multi-model held-out evaluation and report-generation pipeline.

        :return None: Delegates to generate_multimodel_window_reports().
        """
        self.generate_multimodel_window_reports()
