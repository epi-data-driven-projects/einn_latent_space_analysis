"""Select, execute, and save one or more original EINN experiments.

The entry point delegates numerical calculations to the original experiment
functions. It never implements ODE simulation, training, or PCA itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse experiment selectors and optional compute-device overrides.

    :param list[str] | None argv: Explicit arguments, or ``None`` for the CLI.
    :return argparse.Namespace: Selected experiment family and requested cases.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=("single", "dual", "multi", "all"),
        required=True, help="Experiment family to run.",
    )
    parser.add_argument(
        "--model", choices=("sir", "sis", "rost", "all"),
        default="all", help="Single-model epidemiological system.",
    )
    parser.add_argument(
        "--pair", choices=("sir-sis", "sir-seir", "all"),
        default="all", help="Pair of models sharing the Feature module.",
    )
    parser.add_argument(
        "--dim", choices=("5", "20", "all"),
        default="all", help="Feature embedding dimensionality.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Optional PyTorch device override, such as cpu or cuda:1.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print selections without initializing or training models.",
    )
    return parser.parse_args(argv)


def selections(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """Expand a command-line selector into individual experiment runs.

    Dual inference retains the originally specified 20-dimensional embedding
    even when a different dimension is selected for other experiment families.

    :param argparse.Namespace args: Parsed user-defined experiment selectors.
    :return list[tuple[str, str, int]]: Family, model/pair, and embedding size.
    """
    families = (
        ("single", "dual", "multi")
        if args.experiment == "all" else (args.experiment,)
    )
    models = ("sir", "sis", "rost") if args.model == "all" else (args.model,)
    pairs = (
        ("sir-sis", "sir-seir") if args.pair == "all" else (args.pair,)
    )
    dimensions = (5, 20) if args.dim == "all" else (int(args.dim),)
    selected: list[tuple[str, str, int]] = []

    for family in families:
        if family == "single":
            selected.extend(
                (family, model, dimension)
                for model in models
                for dimension in dimensions
            )
        elif family == "dual":
            # The original dual-inference experiment uses d_e=20 exclusively.
            selected.append((family, "sir", 20))
        else:
            selected.extend(
                (family, pair, dimension)
                for pair in pairs
                for dimension in dimensions
            )
    return selected


def run_selected_experiment(
    family: str,
    identifier: str,
    dimension: int,
    device: str | None = None,
) -> None:
    """Execute one existing numerical pipeline and persist its visual reports.

    :param str family: Single-model, dual-inference, or multi-model family.
    :param str identifier: Mechanistic model name or pair identifier.
    :param int dimension: Latent embedding dimensionality to use.
    :param str | None device: Optional PyTorch device override.
    :return None: Original checkpoints, arrays, and new figures are written.
    """
    if family == "single":
        from experiments import single_model_experiment as experiment
        from entry_points.plot_saved import redraw_single

        if device is not None:
            experiment.DEVICE = device
        experiment.run_generalization_experiment(identifier.upper(), dimension)
        saved_directory = (
            Path(experiment.BASE_OUTPUT_DIR) / identifier / f"de_{dimension}"
        )
        redraw_single(saved_directory, identifier.upper(), dimension)

    elif family == "dual":
        if identifier != "sir" or dimension != 20:
            raise ValueError("Original dual inference requires SIR and d_e=20.")
        from experiments import dual_inference_experiment as experiment

        if device is not None:
            experiment.DEVICE = device
        experiment.run_sir_dual_inference_experiment()

    elif family == "multi":
        from experiments import multi_model_experiment as experiment

        experiment.MODEL_NAMES = tuple(
            name.upper() for name in identifier.split("-")
        )
        experiment.EMBEDDING_DIM = dimension
        if device is not None:
            experiment.DEVICE = device
        experiment.OUTPUT_ROOT = (
            f"./outputs/{identifier.replace('-', '_')}_generalization_4x4"
        )
        experiment.run_experiment()

    else:
        raise ValueError(f"Unsupported experiment family: {family}")


def main(argv: list[str] | None = None) -> None:
    """Run or preview the selected experiments sequentially.

    :param list[str] | None argv: Optional command-line arguments for testing.
    :return None: Selected experiments are executed or displayed in dry-run mode.
    """
    args = parse_arguments(argv)
    selected = selections(args)
    for index, (family, identifier, dimension) in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {family}: {identifier}, d_e={dimension}")
        if not args.dry_run:
            run_selected_experiment(family, identifier, dimension, args.device)


if __name__ == "__main__":
    main()
