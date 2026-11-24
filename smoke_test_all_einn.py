#!/usr/bin/env python3
"""Exercise every refactored EINN experiment with minimal, isolated training.

This script is designed for the refactored EINN repository whose modules are
``experiments.*`` and ``entry_points.plot_saved``. It does not modify the
repository's Python files, pretrained checkpoints, or production outputs.

Run from the repository root::

    python smoke_test_all_einn.py --device cuda:1

The smoke test is structural: one epoch and one repetition of *each* training
phase, tiny synthetic datasets, actual inference, original plotting functions,
and saved-array-only redraw. It does not assess predictive accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PHASES = ("phase_1", "phase_2", "phase_3", "phase_4")
DEFAULT_WINDOWS = (10, 18, 40, 80)


def parse_args() -> argparse.Namespace:
    """Parse public smoke-test settings and private subprocess arguments.

    :return argparse.Namespace: Complete command-line configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--experiment", choices=("single", "dual", "multi", "all"),
                        default="all")
    parser.add_argument("--model", choices=("sir", "sis", "rost", "all"),
                        default="all")
    parser.add_argument("--pair", choices=("sir-sis", "sir-seir", "all"),
                        default="all")
    parser.add_argument("--dim", choices=("5", "20", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--phase-reps", type=int, default=1)
    parser.add_argument("--train-trajectories", type=int, default=2)
    parser.add_argument("--test-trajectories", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=900,
                        help="Maximum seconds allowed for each case.")
    parser.add_argument("--results-dir", type=Path, default=Path("smoke_test_results"))
    parser.add_argument("--rost-step", type=float, default=None,
                        help="Optional RK4 internal step for Röst *test only*; "
                             "when omitted the project's original integrator is used.")
    parser.add_argument("--dry-run", action="store_true")
    # Private arguments: the parent process launches one worker per case.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_family", help=argparse.SUPPRESS)
    parser.add_argument("--_identifier", help=argparse.SUPPRESS)
    parser.add_argument("--_dimension", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_destination", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("epochs", "phase_reps", "train_trajectories",
                "test_trajectories", "batch_size", "timeout"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.rost_step is not None and not (0 < args.rost_step <= 1):
        parser.error("--rost-step must be in (0, 1]")
    return args


def select_cases(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """Build the same eleven-case selection as the production entry point.

    :param argparse.Namespace args: Parsed smoke-test selectors.
    :return list[tuple[str, str, int]]: Selected (family, model/pair, dimension).
    """
    from entry_points.run_experiment import selections

    return selections(args)


def quick_train_config(original_class: type, repetitions: int) -> Callable[..., Any]:
    """Create a constructor proxy that replaces only the repetition counts.

    :param type original_class: Unmodified EINNTrainConfig dataclass type.
    :param int repetitions: Number of complete data passes per training phase.
    :return Callable[..., Any]: Callable constructing a standard training config.
    """
    def build_config(*args: Any, **kwargs: Any) -> Any:
        """Construct one configuration without changing its loss weights.

        :param Any args: Original positional constructor arguments.
        :param Any kwargs: Original keyword constructor arguments.
        :return Any: Genuine configuration with reduced phase repetitions.
        """
        kwargs["reps"] = {phase: repetitions for phase in PHASES}
        config = original_class(*args, **kwargs)
        config.reps = {phase: repetitions for phase in PHASES}
        return config

    return build_config


def patch_runner_epochs(epochs: int, batch_size: int) -> None:
    """Limit fixed-epoch single/dual runner calls without touching source files.

    :param int epochs: Outer training epochs for smoke tests.
    :param int batch_size: Training mini-batch size for smoke tests.
    :return None: The runner is patched in the short-lived worker process.
    """
    from einn_runner import EINNRunner

    original_method = EINNRunner.train_and_save

    def run_quickly(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the genuine runner with smaller computational budgets.

        :param Any self: Active EINNRunner instance.
        :param Any args: Original positional training arguments.
        :param Any kwargs: Original keyword training arguments.
        :return Any: Unchanged return value of the original training method.
        """
        kwargs["epochs"] = epochs
        kwargs["batch_size"] = batch_size
        return original_method(self, *args, **kwargs)

    EINNRunner.train_and_save = run_quickly


def patch_rost_solver(step: float | None) -> None:
    """Optionally stabilize Röst RK4 only within the isolated worker.

    :param float | None step: Optional smaller internal integration step.
    :return None: Synthetic-data ODE integration is optionally monkey-patched.
    """
    if step is None:
        return
    import einn.data.synthetic_data_generator as generator_module

    original_odeint = generator_module.odeint

    def integrate(*args: Any, **kwargs: Any) -> Any:
        """Preserve the requested solver while reducing its internal RK4 step.

        :param Any args: Original positional ODE solver arguments.
        :param Any kwargs: Original keyword ODE solver arguments.
        :return Any: Numerical solution from the genuine solver.
        """
        if kwargs.get("method") == "rk4":
            options = dict(kwargs.get("options") or {})
            options["step_size"] = step
            kwargs["options"] = options
        return original_odeint(*args, **kwargs)

    generator_module.odeint = integrate
    print(f"NOTE: Röst synthetic-data integration uses RK4 step_size={step} "
          "only for this smoke test.", flush=True)


def check_pngs(directory: Path) -> list[str]:
    """Validate every PNG header, dimensions, and encoded image data.

    Pillow is already a dependency of Matplotlib, so no new package is needed.
    ``verify()`` checks PNG file integrity without decoding large raster arrays.

    :param Path directory: Directory containing figures from one case.
    :return list[str]: Human-readable problems with missing or invalid images.
    """
    from PIL import Image

    images = sorted(directory.rglob("*.png"))
    if not images:
        return ["No PNG figure was generated."]
    issues: list[str] = []
    for path in images:
        try:
            with Image.open(path) as picture:
                if picture.format != "PNG" or min(picture.size) < 100:
                    issues.append(f"Invalid or tiny PNG: {path}")
                    continue
                picture.verify()
        except Exception as exc:
            issues.append(f"Corrupted or unreadable PNG: {path}: {exc}")
    return issues


def check_numeric_npz(path: Path, keys: tuple[str, ...]) -> list[str]:
    """Check presence, shape, and finiteness of critical result arrays.

    :param Path path: Compressed NumPy result file.
    :param tuple[str, ...] keys: Required numeric result-array names.
    :return list[str]: Missing/non-finite result-array warnings or errors.
    """
    import numpy as np

    if not path.is_file():
        return [f"Missing result file: {path}"]
    problems: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            for name in keys:
                if name not in data:
                    problems.append(f"Missing array {name!r} in {path}")
                    continue
                values = data[name]
                if values.size == 0 or not np.isfinite(values).all():
                    problems.append(f"Empty or non-finite array {name!r} in {path}")
    except Exception as exc:
        problems.append(f"Cannot open {path}: {exc}")
    return problems


def check_reference_dynamics(states: Any, n_population_states: int,
                             identifier: str) -> list[str]:
    """Check ground-truth compartment values and population conservation.

    Röst has three extra cumulative counters, excluded from population mass.
    A smoke run does not judge whether its *trained predictions* are accurate.

    :param Any states: Physical ground-truth state array, time by state.
    :param int n_population_states: Number of exclusive population compartments.
    :param str identifier: Model identifier for diagnostics.
    :return list[str]: Issues found in reference ODE trajectories.
    """
    import numpy as np

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < n_population_states:
        return [f"{identifier}: invalid reference state shape {values.shape}."]
    if not np.isfinite(values).all():
        return [f"{identifier}: reference ODE states contain NaN or infinity."]
    compartments = values[:, :n_population_states]
    issues = []
    if float(compartments.min()) < -1e-4:
        issues.append(f"{identifier}: negative physical compartment values.")
    mass = compartments.sum(axis=1)
    if float(np.max(np.abs(mass - mass[0]))) > 1e-3:
        issues.append(f"{identifier}: population mass is not conserved.")
    return issues


def verify_single(directory: Path, dimension: int) -> list[str]:
    """Validate expected single-model checkpoints, arrays, and redraws.

    :param Path directory: Single-model case directory.
    :param int dimension: Requested latent dimension.
    :return list[str]: Problems found in saved results.
    """
    problems = check_numeric_npz(directory / "test_latent_representations.npz", (
        "true_target", "predictions", "time_embeddings", "feature_embeddings",
        "shared_pca_coords", "cumulative_pca_variance",
    ))
    for name in ("einn_time_module.pth", "einn_feature_module.pth",
                 "einn_output_module.pth", "metrics_by_window.csv"):
        if not (directory / name).is_file():
            problems.append(f"Missing file: {directory / name}")
    try:
        import numpy as np
        with np.load(directory / "test_latent_representations.npz") as data:
            if data["feature_embeddings"].shape != (4, len(data["true_target"]), dimension):
                problems.append("Incorrect feature-embedding tensor dimensions.")
            if data["predictions"].shape != (4, len(data["true_target"])):
                problems.append("Incorrect forecast tensor dimensions.")
    except Exception as exc:
        problems.append(f"Could not check single-model tensor dimensions: {exc}")
    original_path = directory / "test_data_and_scaler.npz"
    try:
        import numpy as np
        with np.load(original_path, allow_pickle=False) as original:
            n_states = 15 if directory.parent.name == "rost" else (
                2 if directory.parent.name == "sis" else 3
            )
            problems.extend(check_reference_dynamics(
                original["test_true_latent_states"], n_states,
                directory.parent.name.upper(),
            ))
    except Exception as exc:
        problems.append(f"Could not validate reference ODE trajectory: {exc}")
    if not any((directory / "ujrarajzolt_abrak").glob("*.png")):
        problems.append("Saved-data-only single-model redraw did not produce PNG.")
    return problems


def verify_dual(directory: Path) -> list[str]:
    """Validate the saved dual-inference panels and replay artifacts.

    :param Path directory: Dual-inference case directory.
    :return list[str]: Problems found in saved results.
    """
    problems = check_numeric_npz(directory / "dual_inference_saved_arrays.npz", (
        "true_target_1", "true_target_2", "pca_common_1_0", "pca_a_1_0",
    ))
    try:
        import numpy as np
        with np.load(directory / "dual_inference_test_inputs.npz", allow_pickle=False) as inputs:
            for prefix in ("high_beta", "low_beta"):
                problems.extend(check_reference_dynamics(
                    inputs[f"{prefix}_true_latent_states"], 3, prefix,
                ))
    except Exception as exc:
        problems.append(f"Could not validate dual reference trajectories: {exc}")
    for index in range(4):
        if not (directory / f"dual_inference_window_{index}.npz").is_file():
            problems.append(f"Missing dual-inference window {index}.")
    for name in ("einn_time_module.pth", "einn_feature_module.pth",
                 "einn_output_module.pth"):
        if not (directory / name).is_file():
            problems.append(f"Missing file: {directory / name}")
    if not any((directory / "redrawn").glob("*.png")):
        problems.append("Saved-data-only dual-inference redraw did not produce PNG.")
    return problems


def verify_multi(directory: Path, names: tuple[str, str], dimension: int) -> list[str]:
    """Validate both model-specific reports, joint embeddings, and redraws.

    :param Path directory: Multi-model report directory.
    :param tuple[str, str] names: Selected pair of mechanistic model identifiers.
    :param int dimension: Requested latent dimension.
    :return list[str]: Missing or invalid result artifacts.
    """
    problems = check_numeric_npz(directory / "all_heldout_feature_embeddings.npz", (
        "feature_embeddings", "true_labels", "trajectory_ids", "time_indices",
    ))
    report_files = list(directory.glob("multimodel_generalization_*_4x4.npz"))
    if len(report_files) < 2:
        problems.append(f"Expected two model-specific 4x4 NPZ files, found {len(report_files)}")
    for path in report_files:
        problems += check_numeric_npz(path, ("true_target", "predictions", "feature_embeddings"))
    for name in names:
        match = list(directory.glob(f"multimodel_generalization_*_true_{name.lower()}_*_4x4.png"))
        if not match:
            problems.append(f"Missing original {name} 4x4 figure.")
    if not list((directory / "redrawn").glob("*.png")):
        problems.append("Saved-data-only multi-model redraw did not produce PNG.")
    original_outputs = directory.parent.parent.parent / "training_and_held_out_data.npz"
    if not original_outputs.is_file():
        problems.append("Missing saved training/held-out trajectory arrays.")
    return problems


def run_worker(args: argparse.Namespace) -> int:
    """Run one configuration with a small dataset and four genuine training phases.

    :param argparse.Namespace args: Internal selection and output-directory paths.
    :return int: Process exit code (zero indicates all smoke checks passed).
    """
    import torch
    from entry_points import plot_saved

    family, identifier, dimension = args._family, args._identifier, args._dimension
    case_dir = args._destination.resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    label = f"{family}:{identifier}:de_{dimension}"
    result: dict[str, Any] = {"case": label, "status": "failed", "issues": [],
                              "output_directory": str(case_dir)}
    try:
        if args.device.startswith("cuda"):
            device = torch.device(args.device)
            gpu_count = torch.cuda.device_count()
            if not torch.cuda.is_available() or (device.index or 0) >= gpu_count:
                raise RuntimeError(f"Requested {args.device}; {gpu_count} CUDA GPUs available.")
        torch.set_num_threads(min(4, torch.get_num_threads()))
        patch_runner_epochs(args.epochs, args.batch_size)
        if family == "single":
            from experiments import single_model_experiment as experiment
            experiment.NUM_STATES = args.train_trajectories
            experiment.EPOCHS = args.epochs
            experiment.BATCH_SIZE = args.batch_size
            experiment.DEVICE = args.device
            experiment.BASE_OUTPUT_DIR = str(case_dir)
            experiment.EINNTrainConfig = quick_train_config(
                experiment.EINNTrainConfig, args.phase_reps
            )
            if identifier == "rost":
                patch_rost_solver(args.rost_step)
            experiment.run_generalization_experiment(identifier.upper(), dimension)
            actual_dir = case_dir / identifier / f"de_{dimension}"
            plot_saved.redraw_single(actual_dir, identifier.upper(), dimension)
            problems = verify_single(actual_dir, dimension)
        elif family == "dual":
            from experiments import dual_inference_experiment as experiment
            experiment.NUM_STATES = args.train_trajectories
            experiment.DEVICE = args.device
            experiment.OUTPUT_DIR = str(case_dir)
            experiment.EINNTrainConfig = quick_train_config(
                experiment.EINNTrainConfig, args.phase_reps
            )
            experiment.run_sir_dual_inference_experiment()
            plot_saved.redraw_dual(case_dir)
            problems = verify_dual(case_dir)
        elif family == "multi":
            from experiments import multi_model_experiment as experiment
            pair = tuple(part.upper() for part in identifier.split("-"))
            experiment.MODEL_NAMES = pair
            experiment.EMBEDDING_DIM = dimension
            experiment.TRAIN_STATES_PER_MODEL = args.train_trajectories
            experiment.TEST_STATES_PER_MODEL = args.test_trajectories
            experiment.EPOCHS = args.epochs
            experiment.BATCH_SIZE = args.batch_size
            experiment.DEVICE = args.device
            experiment.OUTPUT_ROOT = str(case_dir)
            experiment.EINNTrainConfig = quick_train_config(
                experiment.EINNTrainConfig, args.phase_reps
            )
            experiment.run_experiment()
            actual_dir = case_dir / "label_conditioned" / "raw" / f"de_{dimension}"
            plot_saved.redraw_multi(actual_dir, pair, dimension)
            problems = verify_multi(actual_dir, pair, dimension)
        else:
            raise ValueError(f"Unsupported smoke-test family: {family}")
        problems.extend(check_pngs(case_dir))
        if problems:
            raise AssertionError("; ".join(problems))
        result["status"] = "passed"
        result["png_count"] = len(list(case_dir.rglob("*.png")))
        result["npz_count"] = len(list(case_dir.rglob("*.npz")))
        print(f"PASS {label}: {result['png_count']} PNG, {result['npz_count']} NPZ", flush=True)
        return 0
    except Exception as exc:
        result["issues"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        (case_dir / "smoke_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def make_contact_sheet(results: list[dict[str, Any]], destination: Path) -> Path | None:
    """Save a lightweight overview of generated image files for visual review.

    :param list[dict[str, Any]] results: All smoke-test case status objects.
    :param Path destination: Directory in which to save the contact sheet.
    :return Path | None: Contact-sheet PNG, or None if no figures are present.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    thumbnails: list[tuple[str, Path]] = []
    for result in results:
        if result.get("status") != "passed":
            continue
        root = Path(result["output_directory"])
        images = [p for p in root.rglob("*.png") if "redrawn" not in p.parts
                  and "ujrarajzolt_abrak" not in p.parts]
        if not images:
            images = list(root.rglob("*.png"))
        for file in images[:3 if result["case"].startswith("multi:") else 1]:
            thumbnails.append((result["case"], file))
    if not thumbnails:
        return None

    columns = 3
    rows = math.ceil(len(thumbnails) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4.1 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (label, path) in zip(axes.flat, thumbnails):
        try:
            with Image.open(path) as picture:
                picture.thumbnail((720, 520))
                thumbnail = picture.convert("RGB").copy()
            axis.imshow(thumbnail)
            axis.set_title(f"{label}\n{path.name}", fontsize=9)
        except Exception as exc:
            axis.text(0.5, 0.5, f"Cannot preview:\n{exc}", ha="center")
    figure.tight_layout()
    output = destination / "all_figures_overview.png"
    figure.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output


def run_controller(args: argparse.Namespace) -> int:
    """Execute all selections in isolated processes, collecting failures.

    :param argparse.Namespace args: Experiment selectors and smoke-test options.
    :return int: Nonzero if at least one requested case failed its checks.
    """
    cases = select_cases(args)
    if args.dry_run:
        for index, (family, identifier, dimension) in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {family}: {identifier}, d_e={dimension}")
        return 0

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = (args.results_dir / f"run_{now}_{os.getpid()}").resolve()
    destination.mkdir(parents=True, exist_ok=False)
    print(f"Smoke tests: {len(cases)} cases, {args.epochs} epoch(s), "
          f"{args.phase_reps} rep(s) per phase; results: {destination}", flush=True)

    results = []
    script = Path(__file__).resolve()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")

    for index, (family, identifier, dimension) in enumerate(cases, start=1):
        name = f"{family}_{identifier.replace('-', '_')}_de{dimension}"
        case_dir = destination / name
        case_dir.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(script), "--_worker", "--_family", family,
                   "--_identifier", identifier, "--_dimension", str(dimension),
                   "--_destination", str(case_dir), "--device", args.device,
                   "--epochs", str(args.epochs), "--phase-reps", str(args.phase_reps),
                   "--train-trajectories", str(args.train_trajectories),
                   "--test-trajectories", str(args.test_trajectories),
                   "--batch-size", str(args.batch_size)]
        if args.rost_step is not None:
            command.extend(("--rost-step", str(args.rost_step)))
        print(f"\n[{index}/{len(cases)}] START {name}", flush=True)
        log_path = case_dir / "execution.log"
        timed_out = False
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command, cwd=script.parent, env=env,
                    stdout=handle, stderr=subprocess.STDOUT,
                    timeout=args.timeout, check=False,
                )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
        metadata_path = case_dir / "smoke_result.json"
        if metadata_path.exists():
            result = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            result = {"case": f"{family}:{identifier}:de_{dimension}",
                      "output_directory": str(case_dir), "status": "failed", "issues": []}
        if return_code != 0:
            result["status"] = "failed"
            if timed_out:
                result.setdefault("issues", []).append(
                    f"Timed out after {args.timeout} seconds."
                )
            if not result.get("issues"):
                result["issues"] = [f"Worker returned exit code {return_code}."]
        result["log_file"] = str(log_path)
        results.append(result)
        status = "PASS" if result["status"] == "passed" else "FAIL"
        print(f"[{index}/{len(cases)}] {status} {name}", flush=True)
        if status == "FAIL":
            print(f"  Error: {result.get('issues')}", flush=True)
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                print("  Final log lines:\n    " + "\n    ".join(lines[-12:]), flush=True)
            except OSError:
                pass

        summary = {"settings": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()
                                if not key.startswith("_")},
                   "results": results}
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("case", "status", "png_count",
                                                        "npz_count", "issues", "log_file"))
            writer.writeheader()
            for item in results:
                writer.writerow({key: "; ".join(item.get(key, [])) if key == "issues"
                                 else item.get(key, "") for key in writer.fieldnames})

    try:
        preview = make_contact_sheet(results, destination)
        if preview:
            print(f"Figure overview: {preview}", flush=True)
    except Exception as exc:
        print(f"WARNING: Could not create overview: {exc}", flush=True)

    successes = sum(item["status"] == "passed" for item in results)
    print(f"\nFINAL: {successes}/{len(results)} tests passed", flush=True)
    print(f"Summary: {destination / 'summary.csv'}", flush=True)
    return 0 if successes == len(results) else 1


def main() -> int:
    """Choose the controller or one isolated worker process.

    :return int: Zero on full success, otherwise a failure status.
    """
    args = parse_args()
    if args._worker:
        return run_worker(args)
    return run_controller(args)


if __name__ == "__main__":
    sys.exit(main())
