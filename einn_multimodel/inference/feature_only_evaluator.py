from __future__ import annotations
import os
import csv
import numpy as np
import torch
from scipy.spatial import procrustes, cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from einn_multimodel.inference.inference_engine import InferenceEngine
import importlib

class FeatureOnlyEvaluator:
    def __init__(self, models, dataset, metrics, scaling_stats, device, output_dir, target_scaler):
        """Initialize FeatureOnlyEvaluator to store the configuration and initialize internal state.


        :param Any models: The models containing the time, feature, output, and ODE modules.
        :param Any dataset: The prepared padded expanding-window dataset.
        :param Any metrics: Training, forecast, or latent-space diagnostic measurements.
        :param Any scaling_stats: Statistics fitted on training data for physical-domain conversion.
        :param Any device: PyTorch device used for computations.
        :param Any output_dir: Directory where generated results are written.
        :param Any target_scaler: Target normalizer fitted exclusively on training observations.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.experiment = importlib.import_module("experiments.multi_model_experiment")
        self.models = models
        self.dataset = dataset
        self.metrics = metrics
        self.scaling_stats = scaling_stats
        self.target_scaler = target_scaler
        if target_scaler is None or not hasattr(target_scaler, "inverse_transform"):
            raise ValueError("Pass the training-fitted y_scaler.")
        self.device = device
        self.output_dir = output_dir
        self.id_to_model = {v: k for k, v in models.model_to_id.items()}
        self.class_ids = sorted(self.id_to_model)
        target_indices = {model.target_idx for model in models.ode_models.values()}
        if len(target_indices) != 1:
            raise ValueError("All models must agree on the target compartment index.")
        self.target_idx = next(iter(target_indices))
        if not np.allclose(
            [float(scaling_stats.aux_mean[self.target_idx]),
             float(scaling_stats.aux_std[self.target_idx])],
            [float(target_scaler.mean_[0]), float(target_scaler.scale_[0])],
            rtol=1e-3, atol=1e-6,
        ):
            raise ValueError("Incompatible train-fitted AUX and target scaling.")
        os.makedirs(output_dir, exist_ok=True)

    def scale_time(self, t):
        """Map the physical simulation time grid into the network’s normalized domain.


        :param Any t: Time coordinates of the modeled trajectory.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        minimum = self.scaling_stats.t_min.to(self.device)
        maximum = self.scaling_stats.t_max.to(self.device)
        return (t - minimum) / (maximum - minimum).clamp_min(1e-8)

    def clean_target_physical(self, index: int) -> np.ndarray:
        """Convert target observations into valid epidemiological physical values.


        :param int index: Zero-based index of the selected batch element or trajectory.
        :return np.ndarray: Computed result in the configured output representation.
        """
        scaled = np.asarray(self.dataset.y_list[index], dtype=np.float64).reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled).reshape(-1)

    def get_time_module_for_label(self, true_id: int):
        """Return the correct Time Module for the TRUE model label.

        If your local object names differ, most likely ONLY this function needs editing.
        Expected structure: models.time_modules is a dict keyed by model name.

        :param int true_id: Integer identifier of the known data-generating model family.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
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

    def collect_full_trajectories(self, all_trajectories: bool = False):
        """Gather all held-out trajectories for a shared embedding-space analysis.


        :param bool all_trajectories: Collection of independent held-out trajectories for joint evaluation.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
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
                x = torch.as_tensor(self.dataset.x_list[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                t = torch.as_tensor(self.dataset.t_list[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                true_id = int(self.dataset.true_model_labels[idx])
                labels = torch.tensor([true_id], dtype=torch.long, device=self.device)
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

    def evaluate_hausdorff_test_embeddings(self):
        """Compute pairwise geometric distances in original latent dimensions.


        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        data = self.collect_full_trajectories(all_trajectories=True)
        E = data["feature_embeddings"]
        a_id, b_id = (self.models.model_to_id[name] for name in self.experiment.MODEL_NAMES)
        true = data["true_labels"]
        a = E[true == a_id]
        b = E[true == b_id]
        stats = self.experiment.hausdorff_statistics(a, b)
        a_unit = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        b_unit = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        normalized_stats = self.experiment.hausdorff_statistics(a_unit, b_unit)
        return {
            **stats,
            "hausdorff_unit_normalized": normalized_stats["hausdorff"],
            "hausdorff_p95_unit_normalized": normalized_stats["hausdorff_p95"],
        }

    def evaluate_prefix_on_trajectory(self, dataset_idx: int, t_end: int) -> dict:
        """Infer a complete trajectory from one observed historical prefix.


        :param int dataset_idx: Index of the selected sample in the expanding-window dataset.
        :param int t_end: Number of historical time coordinates exposed to the Feature module.
        :return dict: Named tensors, metrics, or model components produced by this operation.
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
        labels = torch.tensor([true_id], dtype=torch.long, device=self.device)

        x_prefix = torch.as_tensor(x_full[:t_end], dtype=torch.float32, device=self.device).unsqueeze(0)
        full_t = torch.as_tensor(t_full, dtype=torch.float32, device=self.device).unsqueeze(0)
        scaled_t = self.scale_time(full_t)

        with torch.no_grad():
            states = InferenceEngine(self.models).predict_future(
                historical_x=x_prefix,
                historical_x_mask=None,
                future_t=full_t,
                scaling_stats=self.scaling_stats,
                true_model_label=labels,
            )
            pred = states[0, :, self.target_idx].detach().cpu().numpy()

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

    def fit_shared_pca_for_phase_results(self, phase_results: list[dict]) -> PCA:
        """Fit one PCA basis to the embeddings of all observation windows.


        :param list[dict] phase_results: Predictions and embedding arrays for every selected observation window.
        :return PCA: Computed result in the configured output representation.
        """
        all_features = np.concatenate([result["latent_feature"] for result in phase_results], axis=0)
        shared = PCA(n_components=2)
        shared.fit(all_features)
        for result in phase_results:
            features = result["latent_feature"]
            result["pca_coords"] = shared.transform(features)
            phase_pca = PCA(n_components=min(features.shape))
            phase_pca.fit(features)
            cumulative = np.cumsum(phase_pca.explained_variance_ratio_)
            result["cum_variance"] = cumulative
            result["pca_95"] = int(np.searchsorted(cumulative, 0.95) + 1)
            result["pca_99"] = int(np.searchsorted(cumulative, 0.99) + 1)
        return shared

    def plot_multimodel_generalization_4x4(self, *args, **kwargs):
        """Delegate rendering to the dedicated multi-model visualization class.

        :param tuple args: Positional plotting arguments.
        :param dict kwargs: Named plotting arguments.
        :return Any: Paths and metrics from the original plotting routine.

        :param tuple args: Additional positional arguments.
        """
        from einn_multimodel.visualization.multi_model_plotter import MultiModelPlotter
        return getattr(MultiModelPlotter(self), "plot_multimodel_generalization_4x4")(*args, **kwargs)

        _save_plot(fig, output_path)

    def plot_joint_latent_space_hausdorff(self, *args, **kwargs):
        """Delegate rendering to the dedicated multi-model visualization class.

        :param tuple args: Positional plotting arguments.
        :param dict kwargs: Named plotting arguments.
        :return Any: Paths and metrics from the original plotting routine.

        :param tuple args: Additional positional arguments.
        """
        from einn_multimodel.visualization.multi_model_plotter import MultiModelPlotter
        return getattr(MultiModelPlotter(self), "plot_joint_latent_space_hausdorff")(*args, **kwargs)

        return destination

    def generate_multimodel_window_reports(self):
        # This is the SAME full-trajectory held-out representation used by the
        # earlier Hausdorff report, not training windows and not pooled vectors.
        """Render a four-by-four diagnostic report for each selected test model.


        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        all_test_data = self.collect_full_trajectories(all_trajectories=True)
        E = all_test_data["feature_embeddings"]
        truth = all_test_data["true_labels"]
        a_id, b_id = (self.models.model_to_id[name] for name in self.experiment.MODEL_NAMES)
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
        # Additional, existing-style embedding export for reproducibility.
        np.savez_compressed(
            os.path.join(self.output_dir, "all_heldout_feature_embeddings.npz"),
            **all_test_data,
        )

        selected = []
        for true_id in self.class_ids:
            indices = [i for i, tr in enumerate(self.dataset.true_model_labels)
                       if int(tr) == true_id]
            if not indices:
                raise RuntimeError(f"No held-out trajectory for model id {true_id}.")
            selected.append(indices[0])

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

    def generate_all_reports(self):
        """Evaluate held-out trajectories and save all requested report artifacts.


        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
        """
        self.generate_multimodel_window_reports()
