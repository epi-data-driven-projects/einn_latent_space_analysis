"""
Load persisted single-model EINN test artifacts and compute latent-space diagnostics.

This module reads saved NumPy archives (.npz) produced by single-model evaluation runs
and computes the geometric and forecasting metrics analyzed in Sections 2.5 and 4.1 of the
accompanying thesis (shared 2D PCA trajectories, per-window cumulative explained variance,
pointwise Time-Feature cosine similarity, Procrustes disparity, and out-of-window MSE)
without retraining or re-running inference.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

from einn.model.interface.saved_experiment import Experiment

# Default directory layout and experiment sweep settings for single-model generalization runs.
DEFAULT_ROOT = Path("./experiment_results/generalization")
RUN_ALL_MODELS = True
ALL_MODEL_NAMES = ("SIR", "SIS", "ROST")
ALL_EMBEDDING_DIMS = (20, 5)
OUTPUT_SUBDIRECTORY = "ujrarajzolt_abrak"

# Default diagnostic and plotting configuration (Section 4.1 uses 4 windows of length 10, 18, 40, 80).
DEFAULT_PCA_COMPONENTS = 8
DEFAULT_OBSERVATION_WINDOWS = (10, 18, 40, 80)
PNG_DPI = 230

# Shared color palette for single-model trajectory, PCA, and cosine-similarity figures.
PALETTE = {
    "ink": "#18283D",
    "muted": "#607185",
    "grid": "#D8E1EB",
    "teal": "#008E97",
    "teal_dark": "#006A76",
    "blue": "#416BA8",
    "navy": "#25415D",
    "orange": "#D78A30",
    "magenta": "#AB4C77",
    "gray": "#8192A6",
    "background_obs": "#E5F0F4",
}

# Distinct bar colors for the 4 observation windows in the Procrustes disparity panel.
PROCRUSTES_COLORS = (
    "#3B7A57",  # Forest green, deliberately distinct from the main report palette.
    "#6B4E9B",  # Deep purple, deliberately distinct from the main report palette.
    "#C44536",  # Muted vermilion, deliberately distinct from the main report palette.
    "#8C6D46",  # Warm bronze, deliberately distinct from the main report palette.
)


def _choose_file(directory: Path) -> tuple[str, Path | list[Path]]:
    """Select the preferred persisted evaluation artifact in a run directory.

    Prefers the consolidated test_latent_representations.npz archive; if absent, falls
    back to sorted per-window temporary snapshots (ablak_*_ideiglenes_eredmenyek.npz).

    :param Path directory: Run directory expected to contain saved .npz evaluation outputs.
    :return tuple[str, Path | list[Path]]: Pair (kind, source) where kind is
        "complete" with a single :class:`Path`, or "temporary" with a sorted list of
        snapshot :class:`Path` objects.
    :raises FileNotFoundError: If neither a complete archive nor temporary snapshots exist.
    """
    full = directory / "test_latent_representations.npz"
    if full.is_file():
        return "complete", full

    # Fall back to partial per-window checkpoints saved during interrupted or incremental runs.
    parts = sorted(directory.glob("ablak_*_ideiglenes_eredmenyek.npz"))
    if parts:
        return "temporary", parts

    raise FileNotFoundError(
        f"Hiányoznak a mentett kiértékelési fájlok: {directory}\n"
        "Várt bemenet: test_latent_representations.npz vagy "
        "ablak_*_ideiglenes_eredmenyek.npz."
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pointwise cosine similarity between aligned Time and Feature embeddings.

    Evaluates directional alignment at each timestep t between matching latent vectors
    (thesis Section 2.5). Timesteps where either vector has near-zero norm (<= 1e-12)
    yield NaN to prevent division by zero.

    :param np.ndarray a: First latent trajectory of shape (T, d_e) (e.g. Time embeddings E_T).
    :param np.ndarray b: Second latent trajectory of shape (T, d_e) (e.g. Feature embeddings E_F).
    :return np.ndarray: Per-timestep cosine similarities of shape (T,) in [-1, 1] (float64).
    """
    # Compute row-wise dot product <a_t, b_t> across the embedding dimension d_e.
    numerator = np.einsum("td,td->t", a, b)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)

    # Guard against degenerate zero-norm vectors by pre-filling the output with NaN.
    return np.divide(
        numerator, denominator,
        out=np.full(a.shape[0], np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )


def _procrustes(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Procrustes disparity between two pointwise-aligned latent trajectories.

    Measures shape dissimilarity D_Proc after optimal translation, uniform scaling, and
    orthogonal rotation/reflection (thesis Section 2.5). Returns NaN if either centered
    trajectory collapses to a single point (Frobenius norm <= 1e-12).

    :param np.ndarray a: Reference latent trajectory of shape (T, d_e).
    :param np.ndarray b: Target latent trajectory of shape (T, d_e) aligned by timestep.
    :return float: Non-negative Procrustes disparity scalar (lower indicates closer geometric match).
    """
    # Check for degenerate constant trajectories before invoking SciPy's Procrustes routine.
    a0 = a - np.mean(a, axis=0)
    b0 = b - np.mean(b, axis=0)
    if np.linalg.norm(a0) <= 1e-12 or np.linalg.norm(b0) <= 1e-12:
        return float("nan")

    _, _, disparity = procrustes(a, b)
    return float(disparity)


def _load_completed(path: Path) -> dict[str, np.ndarray]:
    """Load all evaluation arrays from a consolidated .npz archive.

    :param Path path: Path to test_latent_representations.npz.
    :return dict[str, np.ndarray]: Dictionary containing deep-copied arrays with keys
        "observation_windows" (4,), "true_target" (T,), "predictions"
        (4, T), "time_embeddings" (4, T, d_e), "feature_embeddings"
        (4, T, d_e), and optionally "cosine" (4, T).
    :raises ValueError: If any required array key is missing from the archive.
    """
    with np.load(path, allow_pickle=False) as saved:
        required = (
            "observation_windows", "true_target", "predictions",
            "time_embeddings", "feature_embeddings",
        )
        missing = [key for key in required if key not in saved]
        if missing:
            raise ValueError(f"{path}: hiányzó kulcs(ok): {missing}")

        # Copy arrays out of the NpzFile context so the file handle can close cleanly.
        result = {key: np.array(saved[key], copy=True) for key in required}
        if "cosine_similarities" in saved:
            result["cosine"] = np.array(saved["cosine_similarities"], copy=True)
    return result


def _load_temporary(files: list[Path]) -> dict[str, np.ndarray]:
    """Assemble four per-window temporary .npz snapshots into a unified result dict.

    Validates that all snapshots share the exact same noise-free reference trajectory
    true_target, have distinct observation window lengths, and cover exactly 4 windows
    (sorted in ascending order of observed_steps).

    :param list[Path] files: Paths to per-window snapshot archives (ablak_*_ideiglenes_eredmenyek.npz).
    :return dict[str, np.ndarray]: Stacked arrays matching the schema of :func:`_load_completed`:
        "observation_windows" (4,), "true_target" (T,), "predictions"
        (4, T), "time_embeddings" (4, T, d_e), "feature_embeddings"
        (4, T, d_e), and optionally "cosine" (4, T) if present in all snapshots.
    :raises ValueError: If required keys are missing, targets mismatch across snapshots,
        window lengths are duplicated, or the number of snapshots is not 4.
    """
    results = []
    target = None

    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            required = ("observed_steps", "true_target", "prediction", "time_embedding", "feature_embedding")
            missing = [key for key in required if key not in saved]
            if missing:
                raise ValueError(f"{path}: hiányzó kulcs(ok): {missing}")

            obs = int(np.asarray(saved["observed_steps"]).item())
            current_target = np.asarray(saved["true_target"], dtype=np.float64).reshape(-1).copy()

            # Ensure every window snapshot evaluates the exact same ground-truth test trajectory.
            if target is None:
                target = current_target
            elif not np.allclose(target, current_target, rtol=1e-8, atol=1e-10):
                raise ValueError("A mentett ablakok eltérő teszttrajektóriákat tartalmaznak.")

            entry = {
                "obs": obs,
                "pred": np.asarray(saved["prediction"], dtype=np.float64).reshape(-1).copy(),
                "time": np.array(saved["time_embedding"], copy=True),
                "feature": np.array(saved["feature_embedding"], copy=True),
            }
            if "cosine_similarity" in saved:
                entry["cosine"] = np.asarray(saved["cosine_similarity"], dtype=np.float64).reshape(-1).copy()
            results.append(entry)

    # Order snapshots by ascending observation window length before stacking along axis 0.
    results.sort(key=lambda x: x["obs"])
    observed = [item["obs"] for item in results]
    if len(set(observed)) != len(observed):
        raise ValueError(f"Ismétlődő megfigyelési ablakok: {observed}")
    if len(results) != 4:
        raise ValueError(f"Négy megfigyelési ablak kell, jelenleg {len(results)}: {observed}")

    result = {
        "observation_windows": np.asarray(observed, dtype=int),
        "true_target": target,
        "predictions": np.stack([x["pred"] for x in results]),
        "time_embeddings": np.stack([x["time"] for x in results]),
        "feature_embeddings": np.stack([x["feature"] for x in results]),
    }
    if all("cosine" in x for x in results):
        result["cosine"] = np.stack([x["cosine"] for x in results])
    return result


def load_experiment(directory: Path, model: str, dimension: int) -> Experiment:
    """Load a saved single-model experiment and compute all latent-space diagnostics.

    Validates array shapes and finiteness, projects Feature embeddings onto a single shared
    2D PCA basis across all 4 observation windows (thesis Section 2.5 & 4.1), computes
    per-window cumulative explained variance and intrinsic-dimension thresholds (d95,
    d99), Time-Feature cosine similarities, Procrustes disparities, and strictly
    out-of-window forecast MSEs (t >= observed_steps).

    :param Path directory: Run directory containing the persisted .npz artifact(s).
    :param str model: Epidemiological model identifier (e.g. "SIR", "SIS", "ROST").
    :param int dimension: Expected latent embedding dimension d_e (e.g. 20 or 5).
    :return Experiment: Populated :class:`Experiment` dataclass holding raw arrays and
        derived diagnostics (pca_coords of shape (4, T, 2), cumulative_variance
        of shape (4, min(T, d_e)), cosine of shape (4, T), disparity of
        shape (4,), and mse_future of shape (4,)).
    :raises FileNotFoundError: If no valid .npz artifact is found in directory.
    :raises ValueError: If observation windows, tensor shapes, or numerical finiteness checks fail.
    """
    kind, source = _choose_file(directory)
    arrays = _load_completed(source) if kind == "complete" else _load_temporary(source)

    # Normalize loaded arrays to float64/int with explicit expected shapes:
    # observed: (4,), target: (T,), predictions: (4, T), times/features: (4, T, d_e).
    observed = np.asarray(arrays["observation_windows"], dtype=int).reshape(-1)
    target = np.asarray(arrays["true_target"], dtype=np.float64).reshape(-1)
    predictions = np.asarray(arrays["predictions"], dtype=np.float64)
    times = np.asarray(arrays["time_embeddings"], dtype=np.float64)
    features = np.asarray(arrays["feature_embeddings"], dtype=np.float64)

    # Validate observation window monotonicity, dimensional consistency, and finiteness.
    if observed.size != 4 or observed.size != len(set(observed)) or not np.all(np.diff(observed) > 0):
        raise ValueError("A 4 megfigyelési ablaknak szigorúan növekvőnek kell lennie.")
    if times.ndim != 3 or times.shape != features.shape:
        raise ValueError(f"Eltérő Time és Feature beágyazásméretek: {times.shape}, {features.shape}")
    count, horizon, de = times.shape
    if count != 4 or de != dimension or horizon != target.size or predictions.shape != (count, horizon):
        raise ValueError(
            f"Méretellentmondás: time={times.shape}, target={target.shape}, "
            f"pred={predictions.shape}, várt d_e={dimension}"
        )
    if np.any(observed <= 0) or np.any(observed >= horizon):
        raise ValueError("Hibás megfigyelési ablak; 1 <= ablak < T kell.")
    if not (np.isfinite(times).all() and np.isfinite(features).all() and
            np.isfinite(predictions).all() and np.isfinite(target).all()):
        raise ValueError("Nem véges értékek találhatók a mentett adatokban.")

    # Fit one shared 2D PCA basis across all windows: (4, T, d_e) -> (4*T, d_e) -> (4, T, 2).
    # Sharing the basis ensures 2D latent trajectories are directly comparable across windows.
    shared = PCA(n_components=2)
    flat = features.reshape(-1, de)
    pca_coords = shared.fit_transform(flat).reshape(count, horizon, 2)

    # Fit separate PCAs per window to measure individual cumulative explained variance curves,
    # and determine the minimum component counts reaching 95% and 99% variance (Section 4.1).
    cumulative = np.stack([
        np.cumsum(PCA(n_components=min(horizon, de)).fit(x).explained_variance_ratio_)
        for x in features
    ])
    d95 = np.asarray([np.searchsorted(row, .95) + 1 for row in cumulative])
    d99 = np.asarray([np.searchsorted(row, .99) + 1 for row in cumulative])

    # Recompute pointwise Time-Feature cosine similarity (4, T); prefer stored values if consistent.
    cosine = np.stack([_cosine(a, b) for a, b in zip(times, features)])
    if "cosine" in arrays:
        stored = np.asarray(arrays["cosine"], dtype=np.float64)
        if stored.shape == cosine.shape and not np.allclose(stored, cosine, equal_nan=True, rtol=1e-5, atol=1e-7):
            warnings.warn(
                "Az elmentett koszinuszos hasonlóság nem egyezik a beágyazásokból "
                "újraszámolt értékekkel. A mentett értékeket használjuk.",
                stacklevel=2,
            )
        if stored.shape == cosine.shape:
            cosine = stored

    # Compute full-dimensional Procrustes disparity D_Proc for each window: shape (4,).
    disparity = np.asarray([_procrustes(a, b) for a, b in zip(times, features)])

    # Evaluate forecast MSE strictly on the unobserved future horizon t in [end, T) (Section 4.1).
    mse_future = np.asarray([
        np.mean((predictions[i, end:] - target[end:]) ** 2)
        for i, end in enumerate(observed)
    ], dtype=float)

    return Experiment(
        name=model, dimension=dimension, observations=observed,
        target=target, predictions=predictions,
        time_embeddings=times, feature_embeddings=features,
        cosine=cosine, pca_coords=pca_coords,
        cumulative_variance=cumulative,
        shared_variance_ratio=shared.explained_variance_ratio_,
        disparity=disparity, mse_future=mse_future,
        d95=d95, d99=d99,
        source=str(source if kind == "complete" else f"{len(source)} ideiglenes NPZ"),
    )


def find_runs(root: Path, models: list[str] | None, dims: list[int] | None) -> list[tuple[str, int, Path]]:
    """Discover persisted single-model experiment directories matching model and dimension filters.

    Scans <root>/<model_lower>/de_<d> in canonical model order ("SIR", "SIS", "SEIRM",
    "ROST") and keeps only directories containing valid .npz evaluation artifacts.

    :param Path root: Base directory containing per-model experiment subdirectories.
    :param list[str] | None models: Optional list of model names to include (case-insensitive);
        if None, all subdirectories under root matching supported models are scanned.
    :param list[int] | None dims: Optional list of latent dimensions d_e to include;
        defaults to [20, 5] when None.
    :return list[tuple[str, int, Path]]: List of (model_name, embedding_dim, run_dir) tuples
        for all discovered valid runs.
    :raises FileNotFoundError: If root does not exist, no supported models match, or no
        valid .npz artifacts are found.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Az eredménymappa nem létezik: {root.resolve()}")

    # Restrict discovered directories to supported epidemiological models in canonical order.
    names = [x.upper() for x in models] if models else [x.name.upper() for x in root.iterdir() if x.is_dir()]
    names = [n for n in ("SIR", "SIS", "SEIRM", "ROST") if n in set(names)]
    if not names:
        raise FileNotFoundError(f"Nem találtam modelleket a megadott helyen: {root}")

    runs = []
    for model in names:
        model_dir = root / model.lower()
        dimension_list = dims if dims else [20, 5]
        for d in dimension_list:
            ddir = model_dir / f"de_{d}"
            if not ddir.is_dir():
                if models or dims:
                    warnings.warn(f"Nincs ilyen futtatási mappa, kihagyva: {ddir}", stacklevel=2)
                continue
            # Verify that the directory actually contains a loadable complete or temporary NPZ.
            try:
                _choose_file(ddir)
            except FileNotFoundError:
                if models or dims:
                    warnings.warn(f"Nincsenek NPZ-fájlok, kihagyva: {ddir}", stacklevel=2)
                continue
            runs.append((model, int(d), ddir))

    if not runs:
        raise FileNotFoundError(
            f"Nem találtam újrarajzolható NPZ eredményeket itt: {root.resolve()}\n"
            "Például: sir/de_20/test_latent_representations.npz"
        )
    return runs
