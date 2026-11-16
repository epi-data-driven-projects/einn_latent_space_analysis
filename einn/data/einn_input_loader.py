import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch

from einn.data.data_preprocessor import DataPreprocessor
from einn.data.data_scaling_stats import DataScalingStats
from einn.data.expanding_window_loader import ExpandingWindowLoader
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.base_ode_model import BaseODEModel


class EINNInputLoader:
    """
    Handles the generation, caching, preprocessing, and windowing of synthetic
    epidemiological data for the EINN framework.

    This class coordinates the :class:`SyntheticDataGenerator`,
    :class:`DataPreprocessor`, and :class:`ExpandingWindowLoader` to create a
    seamless data pipeline for model training and evaluation.
    """

    def __init__(
            self,
            true_model: BaseODEModel,
            target_idx: int,
            device: str = 'cpu',
            cache_dir: str = "./cache"
    ):
        """
        Initializes the EINN Input Loader.

        :param BaseODEModel true_model: The underlying physical ODE model (e.g., SIR, Röst)
            used to generate the synthetic ground truth data.
        :param int target_idx: The index of the compartment (e.g., Infected) in the ODE
            state vector that represents the primary target variable for the loss function.
        :param str device: The target computing device ('cpu' or 'cuda'). Defaults to 'cpu'.
        :param str cache_dir: The directory path where generated synthetic datasets
            should be stored to avoid redundant computation. Defaults to "./cache".
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
        Generates or loads a cached synthetic dataset, applies Z-score normalization,
        and structures the data into an expanding window format for EINN training.

        If a compatible cached dataset is found in `self.cache_dir`, it will be loaded
        instead of generating new data.

        :param int num_states: The total number of diverse epidemic trajectories
            (states) to generate or load. Defaults to 20.
        :param int seq_len: The sequence length (number of time steps/days) for each
            generated trajectory. Defaults to 200.
        :param int d_p: The number of learnable physical parameters in the ODE model.
            Defaults to 4.
        :param Optional[List[float]] initial_states: The initial values for the ODE
            compartments at t=0. If None, a default initialization is used based on `d_s`.
        :param Optional[Dict[str, Tuple[float, float]]] param_ranges: A dictionary mapping
            parameter names to a tuple of (min, max) uniform distribution bounds for
            sampling biological parameters. If None, a fallback distribution is used.
        :param Optional[Dict[str, Tuple[float, float]]] correlation_ranges: A dictionary
            mapping covariate names to their (min, max) correlation bounds for sampling.
            Defaults to predefined socio-economic correlations.
        :param float noise_std: The standard deviation of the Gaussian noise applied
            to the synthetic observations. Defaults to 0.05.
        :param int window_step: The step size (in days) by which the expanding window
            grows during training. Defaults to 20.
        :param int window_min_days: The minimum size (in days) of the initial expanding
            window. Defaults to 20.
        :param kwargs: Catches any additional keyword arguments (e.g., deprecated
            `calibrator_model` arguments) to ensure backward compatibility.

        :return Tuple[ExpandingWindowLoader, DataScalingStats]:
            A tuple containing:
            - A configured DataLoader instance ready for multiphase training.
            - A dataclass holding the normalization parameters (mean, std) and time bounds.
        """
        d_s = self.true_model.d_s
        if initial_states is None:
            initial_states = [0.99, 0.01] + [0.0] * (d_s - 2)

        if correlation_ranges is None:
            correlation_ranges = {
                'mobility': (-0.9, -0.4), 'temperature': (-0.4, 0.1),
                'testing': (0.5, 0.9), 'stringency': (0.7, 0.95)
            }

        # Embed seq_len in the filename to prevent loading datasets with mismatched sequence lengths
        cache_file = os.path.join(self.cache_dir, f"{num_states}_states_{seq_len}len_target_{self.target_idx}.pt")

        x_train, y_train, aux_train = [], [], []
        generate_data = True

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

                # 1. INITIAL STATES (Kickstarting the epidemic)
                current_initial_states = initial_states.copy()
                if d_s == 18:  # Special initialization for the comprehensive 18-compartment Röst model
                    current_initial_states = [0.0] * d_s
                    current_initial_states[0] = 0.99
                    current_initial_states[3] = 0.01  # Seed infections into the Ip (Pre-symptomatic) compartment

                # 2. PARAMETER SAMPLING (Eliminating hardcoded values)
                if param_ranges is not None and len(param_ranges) == d_p:
                    true_params = [np.random.uniform(*v) for v in param_ranges.values()]
                else:
                    # Safety fallback to ensure generation succeeds if ranges are missing
                    true_params = [np.random.uniform(0.05, 0.4) for _ in range(d_p)]
                    true_params[0] = np.random.uniform(0.4, 0.7)  # Assign a high beta for an initial outbreak

                # 3. CORRELATION SAMPLING
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

                # Extract the pristine physical solution (copy to isolate memory reference)
                aux = data["true_latent_states"].copy()

                # Replace the pristine target trajectory with the noisy observed data
                aux[:, self.target_idx] = data["y_raw"].ravel()

                x_train.append(data["x_raw"])
                y_train.append(data["y_raw"])
                aux_train.append(aux)

            torch.save({"x_train": x_train, "y_train": y_train, "aux_train": aux_train}, cache_file)
            print("Synthetic dataset successfully cached.")

        # Fit preprocessor globally across all trajectories to establish consistent Z-score boundaries
        preprocessor = DataPreprocessor()
        scaled_dict, scaling_stats = preprocessor.fit_transform(
            x_raw=np.concatenate(x_train, axis=0),
            y_raw=np.concatenate(y_train, axis=0),
            aux_raw=np.concatenate(aux_train, axis=0),
            target_idx=self.target_idx,
            initial_states=initial_states,
            time_length=seq_len
        )

        t_unscaled = np.arange(seq_len).reshape(-1, 1)
        x_scaled, y_scaled, aux_scaled, t_list = [], [], [], []

        # Apply the fitted scalers to each individual trajectory separately
        for i in range(num_states):
            x_scaled.append(preprocessor.x_scaler.transform(x_train[i]))
            y_scaled.append(preprocessor.y_scaler.transform(y_train[i]))
            aux_scaled.append(preprocessor.aux_scaler.transform(aux_train[i]))
            t_list.append(t_unscaled)

        # Structure the scaled data into the dynamically expanding window format
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
