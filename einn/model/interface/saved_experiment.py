from dataclasses import dataclass

import numpy as np


@dataclass
class Experiment:
    """Experiment component of the original EINN implementation.

    :param name: Name of the epidemiological model in the selected experiment.
    :param dimension: Number of dimensions in the saved latent representations.
    :param observations: Observed-prefix lengths used to produce the four report rows.
    :param target: Clean synthetic target trajectory in the physical domain.
    :param predictions: Feature-module predicted target trajectories for each observed prefix.
    :param time_embeddings: Time-module representations over the full target horizon.
    :param feature_embeddings: Feature-module representations over the full target horizon.
    :param cosine: Time-aligned cosine similarities between Time and Feature representations.
    :param pca_coords: Time-indexed Feature embeddings projected to the shared 2D PCA basis.
    :param cumulative_variance: Cumulative explained variance of window-specific Feature PCA.
    :param shared_variance_ratio: Proportion of variance captured by shared PCA components.
    :param disparity: Centered and normalized Procrustes discrepancy by observation window.
    :param mse_future: Prediction MSE restricted to the unobserved target horizon.
    :param d95: Smallest number of PCs explaining at least 95 percent of variance.
    :param d99: Smallest number of PCs explaining at least 99 percent of variance.
    :param source: Path of the NPZ source used to construct the saved report.
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
