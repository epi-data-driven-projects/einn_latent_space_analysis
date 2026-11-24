from dataclasses import dataclass

import numpy as np


@dataclass
class Experiment:
    """Experiment component of the original EINN implementation.

    :param str name: Name of the epidemiological model in the selected experiment.
    :param int dimension: Number of dimensions in the saved latent representations.
    :param np.ndarray observations: Observed-prefix lengths used to produce the four report rows.
    :param np.ndarray target: Clean synthetic target trajectory in the physical domain.
    :param np.ndarray predictions: Feature-module predicted target trajectories for each observed prefix.
    :param np.ndarray time_embeddings: Time-module representations over the full target horizon.
    :param np.ndarray feature_embeddings: Feature-module representations over the full target horizon.
    :param np.ndarray cosine: Time-aligned cosine similarities between Time and Feature representations.
    :param np.ndarray pca_coords: Time-indexed Feature embeddings projected to the shared 2D PCA basis.
    :param np.ndarray cumulative_variance: Cumulative explained variance of window-specific Feature PCA.
    :param np.ndarray shared_variance_ratio: Proportion of variance captured by shared PCA components.
    :param np.ndarray disparity: Centered and normalized Procrustes discrepancy by observation window.
    :param np.ndarray mse_future: Prediction MSE restricted to the unobserved target horizon.
    :param np.ndarray d95: Smallest number of PCs explaining at least 95 percent of variance.
    :param np.ndarray d99: Smallest number of PCs explaining at least 99 percent of variance.
    :param np.ndarray str source: Path of the NPZ source used to construct the saved report.
    """
    name: str
    dimension: int
    observations: np.ndarray              # [4]
    target: np.ndarray                    # [T]
    predictions: np.ndarray               # [4,T]
    time_embeddings: np.ndarray           # [4,T,d_e]
    feature_embeddings: np.ndarray        # [4,T,d_e]
    cosine: np.ndarray                    # [4,T]
    pca_coords: np.ndarray                # [4,T,2]
    cumulative_variance: np.ndarray       # [4,d_e]
    shared_variance_ratio: np.ndarray     # [2]
    disparity: np.ndarray                 # [4]
    mse_future: np.ndarray                # [4]
    d95: np.ndarray                       # [4]
    d99: np.ndarray                       # [4]
    source: str
