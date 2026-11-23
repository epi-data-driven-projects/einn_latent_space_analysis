"""Recreate EINN figures using saved numerical arrays, without inference.

This entry point never imports a trainer, model builder, ODE solver, or
synthetic-data generator. It only uses saved trajectories and visualization
classes, and places regenerated reports in separate output directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def redraw_single(
    saved_directory: Path,
    model_name: str,
    embedding_dim: int,
) -> list[Path]:
    """Rebuild the four-by-four single-model report from saved arrays.

    :param Path saved_directory: Directory containing the original NPZ results.
    :param str model_name: Model family (SIR, SIS, or ROST).
    :param int embedding_dim: Original latent-space dimensionality.
    :return list[Path]: Paths to reconstructed figures and numerical summary.
    """
    from einn.analysis.saved_single_loader import load_experiment
    from einn.visualization.single_four_by_four_plotter import (
        SingleModelFourByFourPlotter,
    )

    experiment = load_experiment(saved_directory, model_name, embedding_dim)
    plotter = SingleModelFourByFourPlotter()
    output_directory = saved_directory / "ujrarajzolt_abrak"
    output_directory.mkdir(parents=True, exist_ok=True)
    return [
        plotter.plot_grid(experiment, output_directory, 8),
        plotter.plot_cosine_detail(experiment, output_directory, None),
        plotter.plot_procrustes_detail(experiment, output_directory),
        plotter.save_summary_csv(experiment, output_directory),
    ]


def redraw_dual(saved_directory: Path) -> tuple[str, str]:
    """Rebuild the full dual-inference grid without loading the trained model.

    :param Path saved_directory: Location of the original PCA and forecast NPZs.
    :return tuple[str, str]: PNG and PDF report paths.
    """
    from einn.visualization.dual_inference_plotter import DualInferencePlotter

    saved_directory = Path(saved_directory)
    packed_arrays = saved_directory / "dual_inference_saved_arrays.npz"
    pca_fields = (
        "pca_common_1", "pca_common_2", "pca_a_1",
        "pca_a_2", "pca_b_1", "pca_b_2",
    )

    with np.load(packed_arrays, allow_pickle=False) as data:
        observation_windows = data["observation_windows"].astype(int)
        true_target_a = data["true_target_1"].copy()
        true_target_b = data["true_target_2"].copy()
        embedding_dim = int(data["embedding_dim"])
        projected_arrays = {
            (index, key): data[f"{key}_{index}"].copy()
            for index in range(len(observation_windows))
            for key in pca_fields
        }

    phase_results = []
    for index, observation_end in enumerate(observation_windows):
        window_path = saved_directory / f"dual_inference_window_{index}.npz"
        with np.load(window_path, allow_pickle=False) as data:
            prediction_a = {
                key[2:]: data[key].copy()
                for key in data.files if key.startswith("a_")
            }
            prediction_b = {
                key[2:]: data[key].copy()
                for key in data.files if key.startswith("b_")
            }

        phase_result = {
            "t_end": int(observation_end),
            "results_1": prediction_a,
            "results_2": prediction_b,
        }
        for key in pca_fields:
            phase_result[key] = projected_arrays[index, key]
        phase_results.append(phase_result)

    output_directory = saved_directory / "redrawn"
    output_directory.mkdir(parents=True, exist_ok=True)
    return DualInferencePlotter().plot_dual_inference_results(
        phase_results,
        true_target_a,
        true_target_b,
        str(output_directory),
        embedding_dim,
    )


def redraw_multi(
    saved_directory: Path,
    pair: tuple[str, str],
    embedding_dim: int,
) -> list[Path]:
    """Rebuild multi-model reports and their joint held-out latent projection.

    :param Path saved_directory: Directory of saved multi-model numeric arrays.
    :param tuple[str, str] pair: The compared mechanistic model identifiers.
    :param int embedding_dim: Number of original latent embedding coordinates.
    :return list[Path]: Paths of the regenerated four-by-four and joint figures.
    """
    from scipy.spatial import cKDTree
    from einn_multimodel.visualization.multi_model_plotter import (
        MultiModelPlotter,
    )

    saved_directory = Path(saved_directory)
    output_directory = saved_directory / "redrawn"
    output_directory.mkdir(parents=True, exist_ok=True)

    experiment = SimpleNamespace(
        MODEL_NAMES=pair,
        EMBEDDING_DIM=embedding_dim,
        OBSERVATION_WINDOWS=(10, 18, 40, 80),
        PCA_COMPONENTS_TO_SHOW=8,
    )
    evaluator_context = SimpleNamespace(
        experiment=experiment,
        output_dir=str(output_directory),
        models=SimpleNamespace(
            model_to_id={name: index for index, name in enumerate(pair)}
        ),
        class_ids=list(range(len(pair))),
        id_to_model={index: name for index, name in enumerate(pair)},
    )
    plotter = MultiModelPlotter(evaluator_context)
    written: list[Path] = []

    # Each independently evaluated true-model trajectory receives its own grid.
    for saved_path in sorted(
        saved_directory.glob("multimodel_generalization_*_4x4.npz")
    ):
        with np.load(saved_path, allow_pickle=False) as arrays:
            observation_windows = arrays["observation_windows"].astype(int)
            phase_results = []
            for index, observation_end in enumerate(observation_windows):
                phase_results.append({
                    "t_end": int(observation_end),
                    "true_target": arrays["true_target"].copy(),
                    "predicted_trajectory": arrays["predictions"][index].copy(),
                    "latent_time": arrays["time_embeddings"][index].copy(),
                    "latent_feature": arrays["feature_embeddings"][index].copy(),
                    "cosine_similarity": arrays["cosine_similarities"][index].copy(),
                    "pca_coords": arrays["pca_coords"][index].copy(),
                    "cum_variance": arrays["cumulative_pca_variance"][index].copy(),
                    "pca_95": int(arrays["pca_d95"][index]),
                    "pca_99": int(arrays["pca_d99"][index]),
                    "procrustes_disparity": float(arrays["procrustes"][index]),
                    "mse_future": float(arrays["future_mse"][index]),
                })
            shared_pca = SimpleNamespace(
                explained_variance_ratio_=arrays["shared_pca_variance_ratio"].copy()
            )

        true_model_name = next(
            (name for name in pair if f"_true_{name.lower()}_" in saved_path.stem),
            pair[0],
        )
        for phase_result in phase_results:
            phase_result["true_model_name"] = true_model_name
        output_path = output_directory / f"{saved_path.stem}_redrawn.png"
        plotter.plot_multimodel_generalization_4x4(
            phase_results=phase_results,
            shared_pca=shared_pca,
            output_path=str(output_path),
            hausdorff_metrics={},
        )
        written.append(output_path)

    held_out_path = saved_directory / "all_heldout_feature_embeddings.npz"
    if not held_out_path.is_file():
        return written

    with np.load(held_out_path, allow_pickle=False) as arrays:
        stored = {key: arrays[key].copy() for key in arrays.files}

    # The source evaluator counts samples after ten expanding-window slices.
    # Preserve this label accounting when reconstructing the chart legend.
    sample_labels = []
    for trajectory_id in np.unique(stored["trajectory_ids"]):
        labels = stored["true_labels"][stored["trajectory_ids"] == trajectory_id]
        sample_labels.extend([int(labels[0])] * 10)
    evaluator_context.dataset = SimpleNamespace(
        true_model_labels=np.asarray(sample_labels)
    )

    first = stored["feature_embeddings"][stored["true_labels"] == 0]
    second = stored["feature_embeddings"][stored["true_labels"] == 1]
    if first.ndim != 2 or second.ndim != 2 or not len(first) or not len(second):
        return written

    distances_ab = cKDTree(second).query(first, k=1)[0]
    distances_ba = cKDTree(first).query(second, k=1)[0]
    unit_first = first / np.maximum(
        np.linalg.norm(first, axis=1, keepdims=True), 1e-8
    )
    unit_second = second / np.maximum(
        np.linalg.norm(second, axis=1, keepdims=True), 1e-8
    )
    normalized_ab = cKDTree(unit_second).query(unit_first, k=1)[0]
    normalized_ba = cKDTree(unit_first).query(unit_second, k=1)[0]
    metrics = {
        "hausdorff": float(max(distances_ab.max(), distances_ba.max())),
        "hausdorff_p95": float(max(
            np.percentile(distances_ab, 95), np.percentile(distances_ba, 95)
        )),
        "directed_a_to_b": float(distances_ab.max()),
        "directed_b_to_a": float(distances_ba.max()),
        "hausdorff_unit_normalized": float(max(
            normalized_ab.max(), normalized_ba.max()
        )),
        "hausdorff_p95_unit_normalized": float(max(
            np.percentile(normalized_ab, 95),
            np.percentile(normalized_ba, 95)
        )),
    }
    written.append(
        Path(plotter.plot_joint_latent_space_hausdorff(stored, metrics))
    )
    return written


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse saved-data-only selectors and output roots.

    :param list[str] | None argv: Optional command-line arguments for testing.
    :return argparse.Namespace: Selected models, dimensions, and source folders.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=("single", "dual", "multi", "all"),
        required=True,
    )
    parser.add_argument(
        "--model", choices=("sir", "sis", "rost", "all"), default="all",
    )
    parser.add_argument(
        "--pair", choices=("sir-sis", "sir-seir", "all"), default="all",
    )
    parser.add_argument(
        "--dim", choices=("5", "20", "all"), default="all",
    )
    parser.add_argument(
        "--single-root", type=Path,
        default=Path("experiment_results/generalization"),
    )
    parser.add_argument(
        "--dual-root", type=Path,
        default=Path("experiment_results/sir_dual_inference"),
    )
    parser.add_argument(
        "--multi-root", type=Path, default=Path("outputs"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Regenerate selected graphics from previously serialized trajectories.

    :param list[str] | None argv: Optional argument list for scripted execution.
    :return None: Paths of the regenerated figures are printed to the console.
    """
    args = parse_arguments(argv)
    families = (
        ("single", "dual", "multi")
        if args.experiment == "all" else (args.experiment,)
    )
    dimensions = (5, 20) if args.dim == "all" else (int(args.dim),)
    models = ("sir", "sis", "rost") if args.model == "all" else (args.model,)
    pairs = (
        ("sir-sis", "sir-seir") if args.pair == "all" else (args.pair,)
    )

    for family in families:
        if family == "single":
            for model in models:
                for dimension in dimensions:
                    saved_directory = (
                        args.single_root / model / f"de_{dimension}"
                    )
                    for output_path in redraw_single(
                        saved_directory, model.upper(), dimension
                    ):
                        print(output_path)
        elif family == "dual":
            if args.experiment == "dual" and args.dim not in ("all", "20"):
                raise ValueError("Original dual inference uses d_e=20.")
            print(redraw_dual(args.dual_root))
        else:
            for pair in pairs:
                for dimension in dimensions:
                    root = (
                        args.multi_root
                        / f"{pair.replace('-', '_')}_generalization_4x4"
                    )
                    saved_directory = (
                        root / "label_conditioned" / "raw" / f"de_{dimension}"
                    )
                    for output_path in redraw_multi(
                        saved_directory,
                        tuple(name.upper() for name in pair.split("-")),
                        dimension,
                    ):
                        print(output_path)


if __name__ == "__main__":
    main()
