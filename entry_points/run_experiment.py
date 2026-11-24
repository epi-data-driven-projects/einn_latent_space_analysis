"""
CLI entry point for selecting, running, and persisting EINN experiment pipelines.

Delegates numerical simulation, training, evaluation, and plotting to the underlying
single-model, dual-inference, and multi-model experiment modules without reimplementing
model dynamics or latent-space analyses in the runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI flags for experiment family selection, model/pair/dimension filters, and device.

    :param list[str] | None argv: Optional argument list; defaults to `sys.argv[1:]` when None.
    :return argparse.Namespace: Parsed options controlling experiment expansion, device override,
        and dry-run preview mode.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("single", "dual", "multi", "all"),
        required=True,
        help="Experiment family to run.",
    )
    parser.add_argument(
        "--model",
        choices=("sir", "sis", "rost", "all"),
        default="all",
        help="Single-model epidemiological system.",
    )
    parser.add_argument(
        "--pair",
        choices=("sir-sis", "sir-seir", "all"),
        default="all",
        help="Pair of models sharing the Feature module.",
    )
    parser.add_argument(
        "--dim",
        choices=("5", "20", "all"),
        default="all",
        help="Feature embedding dimensionality.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional PyTorch device override, such as cpu or cuda:1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selections without initializing or training models.",
    )
    return parser.parse_args(argv)


def selections(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """
    Expand CLI filter arguments into an ordered list of `(family, identifier, dimension)` runs.

    Forces `("dual", "sir", 20)` for dual-inference runs because the reference dual-inference
    experiment is defined exclusively at embedding dimension d_e = 20.

    :param argparse.Namespace args: Parsed namespace from `parse_arguments`.
    :return list[tuple[str, str, int]]: Ordered tuples `(family, identifier, embedding_dim)`
        to execute or preview.
    """
    families = (
        ("single", "dual", "multi")
        if args.experiment == "all"
        else (args.experiment,)
    )
    models = ("sir", "sis", "rost") if args.model == "all" else (args.model,)
    pairs = (
        ("sir-sis", "sir-seir")
        if args.pair == "all"
        else (args.pair,)
    )
    dimensions = (5, 20) if args.dim == "all" else (int(args.dim),)
    selected: list[tuple[str, str, int]] = []

    for family in families:
        if family == "single":
            # Cartesian grid over selected single-model systems and embedding dimensions.
            selected.extend(
                (family, model, dimension)
                for model in models
                for dimension in dimensions
            )
        elif family == "dual":
            # Reference dual-inference experiment is fixed to SIR at d_e = 20.
            selected.append(("dual", "sir", 20))
        else:
            # Cartesian grid over selected multi-model pairs and embedding dimensions.
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
    """
    Dispatch and execute a single `(family, identifier, dimension)` experiment pipeline.

    Applies optional device overrides to the target experiment module before running its
    entry function and writes serialized outputs and figures to disk.

    :param str family: Experiment family identifier ('single', 'dual', or 'multi').
    :param str identifier: Lowercase single-model name (e.g. 'sir') or hyphenated pair (e.g. 'sir-sis').
    :param int dimension: Latent embedding dimension d_e for the run.
    :param str | None device: Optional PyTorch device string overriding the module's default DEVICE.
    :return None: Runs the selected experiment and saves checkpoints, arrays, and figures.
    :raises ValueError: If `family` is unsupported or `dual` is invoked with non-`sir` or `dimension != 20`.
    """
    if family == "single":
        from entry_points.plot_saved import redraw_single
        from experiments import single_model_experiment as experiment

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

        # Configure multi-model pair, embedding size, optional device, and output directory root.
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
    """
    Parse arguments and run or dry-run all selected EINN experiments sequentially.

    :param list[str] | None argv: Optional CLI argument list for programmatic or test invocation.
    :return None: Prints progress lines to stdout and runs non-dry selections.
    """
    args = parse_arguments(argv)
    selected = selections(args)
    for index, (family, identifier, dimension) in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {family}: {identifier}, d_e={dimension}")
        if not args.dry_run:
            run_selected_experiment(family, identifier, dimension, args.device)


if __name__ == "__main__":
    main()
