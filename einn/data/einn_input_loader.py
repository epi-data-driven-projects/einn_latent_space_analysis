import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from einn.data.data_preprocessor import DataPreprocessor
from einn.data.data_scaling_stats import DataScalingStats
from einn.data.expanding_window_loader import ExpandingWindowLoader
from einn.data.ode_solution_generator import ODESolutionGenerator
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.base_ode_model import BaseODEModel


class EINNInputLoader:
    """
    Data orchestration class responsible for generating synthetic ground-truth data,
    running ODE pre-calibration, normalizing features (Z-Score), and slicing the data
    into expanding windows before handing them off to the EINN Runner.
    """

    def __init__(
            self,
            true_model: BaseODEModel,
            calibrator_model: BaseODEModel,
            target_idx: int,
            device: str = 'cpu',
            cache_dir: str = "./cache"
    ):
        """
        Initializes the data pipeline orchestrator.

        :param BaseODEModel true_model: The mathematical model used to generate the synthetic ground truth.
        :param BaseODEModel calibrator_model: The mathematical model used by L-BFGS to find auxiliary constraints.
        :param int target_idx: The compartment index representing the observed target (e.g., 2 for Infected).
        :param str device: The target computation device ('cpu' or 'cuda').
        :param str cache_dir: Directory path where generated datasets are saved to prevent re-computation.
        """
        self.true_model = true_model
        self.calibrator_model = calibrator_model
        self.target_idx = target_idx
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def prepare_training_data(
            self,
            num_states: int = 20,
            seq_len: int = 200,
            d_p: int = 4,
            initial_states: List[float] = None,
            param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
            correlation_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
            noise_std: float = 0.02,
            calibrator_epoch: int = 100,
            calibrator_lr: float = 0.001,
            window_step: int = 20,
            window_min_days: int = 20
    ) -> Tuple[ExpandingWindowLoader, DataScalingStats]:
        """
        Generates diverse synthetic states, pre-calibrates auxiliary trajectories, normalizes
        all features via Z-score scaling, and prepares the ExpandingWindowLoader.

        :param int num_states: The number of distinct synthetic parameterizations (regions) to generate.
        :param int seq_len: Total length (in days) of each generated time-series.
        :param int d_p: Number of ODE physical parameters to calibrate (e.g., 4 for SEIRM).
        :param List[float] initial_states: The physical compartment fractions at t=0. Defaults to SEIRM standard.
        :param dict param_ranges: Min/Max bounds for generating ODE parameters (beta, alpha, gamma, mu).
        :param dict correlation_ranges: Min/Max bounds for the correlations of exogenous variables.
        :param float noise_std: Standard deviation of the Gaussian noise applied to the synthetic target data.
        :param int calibrator_epoch: Number of optimization steps for the L-BFGS pre-calibrator.
        :param float calibrator_lr: Learning rate for the pre-calibrator.
        :param int window_step: Expanding step size in days for the dataset slicing.
        :param int window_min_days: Minimum historical days needed for the first window.

        :return Tuple[ExpandingWindowLoader, DataScalingStats]: The prepared PyTorch Dataset and the scaling statistics.
        """
        if initial_states is None:
            initial_states = [0.99, 0.01, 0.0, 0.0, 0.0]

        # Default randomization ranges if not explicitly provided
        if param_ranges is None:
            param_ranges = {
                'beta': (0.3, 0.6), 'alpha': (0.1, 0.3),
                'gamma': (0.05, 0.15), 'mu': (0.01, 0.08)
            }

        if correlation_ranges is None:
            correlation_ranges = {
                'mobility': (-0.9, -0.4), 'temperature': (-0.4, 0.1),
                'testing': (0.5, 0.9), 'stringency': (0.7, 0.95)
            }

        cache_file = os.path.join(self.cache_dir, f"{num_states}_states_target_{self.target_idx}.pt")
        calibrator = ODESolutionGenerator(assumed_ode_model=self.calibrator_model, d_p=d_p, device=self.device)

        x_train, y_train, aux_train = [], [], []
        generate_data = True

        if os.path.exists(cache_file):
            print(f"Found cache: {cache_file}")
            cached_data = torch.load(cache_file, weights_only=False)

            # Check if the cache contains enough states to satisfy the current request
            if len(cached_data["x_train"]) >= num_states:
                print(f"Successfully loaded {num_states} states from cache.")
                x_train = cached_data["x_train"][:num_states]
                y_train = cached_data["y_train"][:num_states]
                aux_train = cached_data["aux_train"][:num_states]
                generate_data = False
            else:
                print(f"Cache only contains {len(cached_data['x_train'])} states. Regenerating data...")

        if generate_data:
            print(f"Generating and pre-calibrating {num_states} diverse states...")
            for i in range(num_states):
                # Dynamically sample true physical parameters based on the provided ranges
                true_params = [
                    np.random.uniform(*param_ranges['beta']),
                    np.random.uniform(*param_ranges['alpha']),
                    np.random.uniform(*param_ranges['gamma']),
                    np.random.uniform(*param_ranges['mu'])
                ]

                # Dynamically sample exogenous feature correlations
                correlations = {
                    k: np.random.uniform(*v) for k, v in correlation_ranges.items()
                }

                # Generate the biological ground truth
                generator = SyntheticDataGenerator(true_ode_model=self.true_model, seq_len=seq_len)
                data = generator.generate(
                    initial_states=initial_states,
                    true_params=true_params,
                    noise_std=noise_std,
                    correlations=correlations
                )

                # Find the best physically plausible curves assuming incomplete knowledge
                aux = calibrator.calibrate(
                    y_raw=data["y_raw"],
                    initial_states=initial_states,
                    epochs=calibrator_epoch,
                    lr=calibrator_lr
                )

                # Anchoring trick: Force the targeted compartment in the auxiliary data to match reality exactly
                aux[:, self.target_idx] = data["y_raw"].ravel()

                x_train.append(data["x_raw"])
                y_train.append(data["y_raw"])
                aux_train.append(aux)

            # Persist generated data to disk to speed up subsequent runs
            torch.save({"x_train": x_train, "y_train": y_train, "aux_train": aux_train}, cache_file)
            print("Synthetic dataset successfully cached.")

        preprocessor = DataPreprocessor()
        scaled_dict, scaling_stats = preprocessor.fit_transform(
            x_raw=np.concatenate(x_train, axis=0),
            y_raw=np.concatenate(y_train, axis=0),
            aux_raw=np.concatenate(aux_train, axis=0),
            target_idx=self.target_idx,
            initial_states=initial_states
        )

        t_unscaled = np.arange(seq_len).reshape(-1, 1)
        x_scaled, y_scaled, aux_scaled, t_list = [], [], [], []

        for i in range(num_states):
            x_scaled.append(preprocessor.x_scaler.transform(x_train[i]))
            y_scaled.append(preprocessor.y_scaler.transform(y_train[i]))
            aux_scaled.append(preprocessor.aux_scaler.transform(aux_train[i]))
            t_list.append(t_unscaled)

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
