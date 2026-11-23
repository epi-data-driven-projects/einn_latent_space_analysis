"""Read persisted single-model test arrays without retraining or inference."""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from einn.model.interface.saved_experiment import Experiment
DEFAULT_ROOT = Path("./experiment_results/generalization")

RUN_ALL_MODELS = True

ALL_MODEL_NAMES = ("SIR", "SIS", "ROST")

ALL_EMBEDDING_DIMS = (20, 5)

OUTPUT_SUBDIRECTORY = "ujrarajzolt_abrak"

DEFAULT_PCA_COMPONENTS = 8

DEFAULT_OBSERVATION_WINDOWS = (10, 18, 40, 80)

PNG_DPI = 230

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

PROCRUSTES_COLORS = (
    "#3B7A57",  # Forest green, deliberately distinct from the main report palette.
    "#6B4E9B",  # Deep purple, deliberately distinct from the main report palette.
    "#C44536",  # Muted vermilion, deliberately distinct from the main report palette.
    "#8C6D46",  # warm bronze
)



def _choose_file(directory: Path) -> tuple[str, Path | list[Path]]:
    """Choose the preferred existing numeric output file from available candidates.


    :param Path directory: Directory containing the saved input data or receiving report files.
    :return tuple[str, Path | list[Path]]: Computed result in the configured output representation.
    """
    full = directory / "test_latent_representations.npz"
    if full.is_file():
        return "complete", full
    parts = sorted(directory.glob("ablak_*_ideiglenes_eredmenyek.npz"))
    if parts:
        return "temporary", parts
    raise FileNotFoundError(
        f"Hiányoznak a mentett kiértékelési fájlok: {directory}\n"
        "Várt bemenet: test_latent_representations.npz vagy "
        "ablak_*_ideiglenes_eredmenyek.npz."
    )

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Evaluate cosine-similarity diagnostics between matching latent embeddings.


    :param np.ndarray a: First operand or trajectory in the numerical comparison.
    :param np.ndarray b: Second operand or trajectory in the numerical comparison.
    :return np.ndarray: Computed result in the configured output representation.
    """
    numerator = np.einsum("td,td->t", a, b)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numerator, denominator,
        out=np.full(a.shape[0], np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )

def _procrustes(a: np.ndarray, b: np.ndarray) -> float:
    """Measure geometric alignment between centered and normalized latent trajectories.


    :param np.ndarray a: First operand or trajectory in the numerical comparison.
    :param np.ndarray b: Second operand or trajectory in the numerical comparison.
    :return float: Computed result in the configured output representation.
    """
    a0 = a - np.mean(a, axis=0)
    b0 = b - np.mean(b, axis=0)
    if np.linalg.norm(a0) <= 1e-12 or np.linalg.norm(b0) <= 1e-12:
        return float("nan")
    _, _, disparity = procrustes(a, b)
    return float(disparity)

def _load_completed(path: Path):
    """Load fully persisted experiment results for visualization.


    :param Path path: File-system path to the experiment artifact or configuration.
    :return Any: The computed objects, metrics, or artifact paths for the requested operation.
    """
    with np.load(path, allow_pickle=False) as saved:
        required = (
            "observation_windows", "true_target", "predictions",
            "time_embeddings", "feature_embeddings",
        )
        missing = [key for key in required if key not in saved]
        if missing:
            raise ValueError(f"{path}: hiányzó kulcs(ok): {missing}")
        result = {key: np.array(saved[key], copy=True) for key in required}
        if "cosine_similarities" in saved:
            result["cosine"] = np.array(saved["cosine_similarities"], copy=True)
    return result

def _load_temporary(files: Sequence[Path]):
    """Load an intermediate experiment snapshot when final results are unavailable.
    
    :param Sequence[Path] files: Candidate saved results to be located and loaded.
    :return Any: The computed objects, metrics, or artifact paths for the requested operation.
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
    """Retrieve load experiment from the configured model or saved dataset.

    :param Path directory: Directory containing the saved input data or receiving report files.
    :param str model: Neural network or epidemiological ODE model being evaluated.
    :param int dimension: Number of latent embedding coordinates used by this experiment.
    :return Experiment: Computed result in the configured output representation.
    """
    kind, source = _choose_file(directory)
    arrays = _load_completed(source) if kind == "complete" else _load_temporary(source)
    observed = np.asarray(arrays["observation_windows"], dtype=int).reshape(-1)
    target = np.asarray(arrays["true_target"], dtype=np.float64).reshape(-1)
    predictions = np.asarray(arrays["predictions"], dtype=np.float64)
    times = np.asarray(arrays["time_embeddings"], dtype=np.float64)
    features = np.asarray(arrays["feature_embeddings"], dtype=np.float64)
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

    # Fit one shared PCA basis to every saved Feature embedding and observation window.
    shared = PCA(n_components=2)
    flat = features.reshape(-1, de)
    pca_coords = shared.fit_transform(flat).reshape(count, horizon, 2)
    cumulative = np.stack([
        np.cumsum(PCA(n_components=min(horizon, de)).fit(x).explained_variance_ratio_)
        for x in features
    ])
    d95 = np.asarray([np.searchsorted(row, .95) + 1 for row in cumulative])
    d99 = np.asarray([np.searchsorted(row, .99) + 1 for row in cumulative])
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

    disparity = np.asarray([_procrustes(a, b) for a, b in zip(times, features)])
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
    """Discover saved experiment runs matching the chosen model and dimension.


    :param Path root: Base directory from which experiment results are loaded.
    :param list[str] | None models: The models containing the time, feature, output, and ODE modules.
    :param list[int] | None dims: Available embedding dimensionalities for the selected runs.
    :return list[tuple[str, int, Path]]: Collection of the generated results in their original order.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Az eredménymappa nem létezik: {root.resolve()}")
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
