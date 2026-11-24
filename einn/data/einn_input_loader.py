"""Synthetic training dataset assembly, disk caching, and preprocessing for single-model EINN."""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from einn.data.data_preprocessor import DataPreprocessor
from einn.data.data_scaling_stats import DataScalingStats
from einn.data.expanding_window_loader import ExpandingWindowLoader
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.base_ode_model import BaseODEModel


class EINNInputLoader:
    """
    Generate, cache, standardize, and package synthetic ODE trajectories for EINN training.

    Implements the synthetic data pipeline described in thesis Section 2.3: uniformly samples
    mechanistic parameters and exogenous-variable correlations across multiple trajectories,
    fits a single Z-score preprocessor on the combined training set (Section 3), and builds
    an ExpandingWindowLoader over nested observation prefixes.
    """

    def __init__(
        self,
        true_model: BaseODEModel,
        target_idx: int,
        device: str = 'cpu',
        cache_dir: str = "./cache"
    ):
        """
        Initialize the loader and ensure the trajectory cache directory exists.

        :param BaseODEModel true_model: Epidemiological ODE model used to simulate reference trajectories.
        :param int target_idx: Compartment index j* in [0, d_s) selected as the observed target variable.
        :param str device: Target PyTorch device string for the resulting ExpandingWindowLoader.
        :param str cache_dir: Directory where generated raw trajectories are cached as .pt files.
        """
        self.true_model = true_model
        self.target_idx = target_idx
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def prepare_training_data(
        self,
        num_states: int = 20,
        seq_len: int = 200,
        d_p: int = 4,
        initial_states: Optional[List[float]] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        correlation_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        noise_std: float = 0.05,
        window_step: int = 20,
        window_min_days: int = 20,
        **kwargs
    ) -> Tuple[ExpandingWindowLoader, DataScalingStats]:
        """
        Load or generate synthetic training trajectories, standardize them, and build the window loader.

        Each trajectory has shape (seq_len, d_x) for inputs, (seq_len, 1) for targets, and
        (seq_len, d_s) for auxiliary reference states. Scalers are fit once on all concatenated
        training trajectories of shape (num_states * seq_len, *), while temporal min-max bounds
        remain fixed to a single trajectory horizon [0, seq_len - 1].

        :param int num_states: Number of independent synthetic trajectories to load or simulate.
        :param int seq_len: Number of timesteps T per trajectory.
        :param int d_p: Number of mechanistic ODE parameters sampled per trajectory.
        :param Optional[List[float]] initial_states: Physical initial compartment proportions s(0)
            of length d_s; defaults to [0.99, 0.01, 0.0, ...] (Section 2.3).
        :param Optional[Dict[str, Tuple[float, float]]] param_ranges: Mapping of parameter names
            to (low, high) uniform sampling intervals; must have length d_p to take effect.
        :param Optional[Dict[str, Tuple[float, float]]] correlation_ranges: Mapping of the four
            synthetic covariates (mobility, temperature, testing, stringency) to (low, high)
            target-correlation intervals rho_i in [-1, 1] (Section 2.3).
        :param float noise_std: Standard deviation of additive Gaussian observation noise.
        :param int window_step: Step size between consecutive expanding-window lengths.
        :param int window_min_days: Shortest observation prefix length used in training.
        :param dict kwargs: Accepted for backward compatibility with legacy callers; ignored.
        :return Tuple[ExpandingWindowLoader, DataScalingStats]: Pair (dataset, scaling_stats)
            containing the standardized expanding-window loader and fitted Z-score/time statistics.
        """
        d_s = self.true_model.d_s
        if initial_states is None:
            initial_states = [0.99, 0.01] + [0.0] * (d_s - 2)

        # Default target-correlation intervals for the 4 synthetic exogenous covariates (Section 2.3).
        if correlation_ranges is None:
            correlation_ranges = {
                'mobility': (-0.9, -0.4), 'temperature': (-0.4, 0.1),
                'testing': (0.5, 0.9), 'stringency': (0.7, 0.95)
            }

        # Encode trajectory count, horizon length, and target index in the cache key to avoid mismatches.
        cache_file = os.path.join(self.cache_dir, f"{num_states}_states_{seq_len}len_target_{self.target_idx}.pt")

        x_train, y_train, aux_train = [], [], []
        generate_data = True

        # Reuse cached raw trajectories if available and large enough for the requested num_states.
        if os.path.exists(cache_file):
            print(f"Found cache: {cache_file}")
            cached_data = torch.load(cache_file, weights_only=False)
            if len(cached_data["x_train"]) >= num_states:
                print(f"Successfully loaded {num_states} states from cache.")
                x_train = cached_data["x_train"][:num_states]
                y_train = cached_data["y_train"][:num_states]
                aux_train = cached_data["aux_train"][:num_states]
                generate_data = False

        if generate_data:
            print(f"Generating {num_states} diverse states (d_s={d_s}, d_p={d_p})...")
            for i in range(num_states):
                current_initial_states = initial_states.copy()
                # For the 18-compartment Röst model, seed 1% of the population in the presymptomatic state I_p (index 3).
                if d_s == 18:
                    current_initial_states = [0.0] * d_s
                    current_initial_states[0] = 0.99
                    current_initial_states[3] = 0.01

                # Sample ODE parameters uniformly from configured bounds, or fall back to generic outbreak ranges.
                if param_ranges is not None and len(param_ranges) == d_p:
                    true_params = [np.random.uniform(*v) for v in param_ranges.values()]
                else:
                    true_params = [np.random.uniform(0.05, 0.4) for _ in range(d_p)]
                    # Boost the transmission parameter so fallback simulations produce a clear epidemic wave.
                    true_params[0] = np.random.uniform(0.4, 0.7)

                # Sample target-correlation coefficients rho_i for each exogenous covariate.
                correlations = {
                    k: np.random.uniform(*v) for k, v in correlation_ranges.items()
                }

                generator = SyntheticDataGenerator(true_ode_model=self.true_model, seq_len=seq_len)
                data = generator.generate(
                    initial_states=current_initial_states,
                    true_params=true_params,
                    noise_std=noise_std,
                    correlations=correlations
                )

                # Copy clean ODE states (seq_len, d_s) and replace the target column with observed y_raw (Section 2.3).
                aux = data["true_latent_states"].copy()
                aux[:, self.target_idx] = data["y_raw"].ravel()

                x_train.append(data["x_raw"])   # Shape: (seq_len, d_x)
                y_train.append(data["y_raw"])   # Shape: (seq_len, 1)
                aux_train.append(aux)           # Shape: (seq_len, d_s)

            torch.save({"x_train": x_train, "y_train": y_train, "aux_train": aux_train}, cache_file)
            print("Synthetic dataset successfully cached.")

        # Fit global Z-score scalers across all concatenated trajectories: (num_states * seq_len, *).
        preprocessor = DataPreprocessor()
        scaled_dict, scaling_stats = preprocessor.fit_transform(
            x_raw=np.concatenate(x_train, axis=0),
            y_raw=np.concatenate(y_train, axis=0),
            aux_raw=np.concatenate(aux_train, axis=0),
            target_idx=self.target_idx,
            initial_states=initial_states,
            time_length=seq_len
        )

        # Transform each trajectory individually using the fitted global scalers and attach time grid (seq_len, 1).
        t_unscaled = np.arange(seq_len).reshape(-1, 1)
        x_scaled, y_scaled, aux_scaled, t_list = [], [], [], []

        for i in range(num_states):
            x_scaled.append(preprocessor.x_scaler.transform(x_train[i]))
            y_scaled.append(preprocessor.y_scaler.transform(y_train[i]))
            aux_scaled.append(preprocessor.aux_scaler.transform(aux_train[i]))
            t_list.append(t_unscaled)

        # Build the expanding-window batch loader over prefix lengths [window_min_days, seq_len] (Section 2.3).
        dataset = ExpandingWindowLoader(
            x_list=x_scaled,
            y_list=y_scaled,
            t_list=t_list,
            aux_list=aux_scaled,
            min_days=window_min_days,
            max_days=seq_len,
            step=window_step,
            device=self.device
        )

        return dataset, scaling_stats
