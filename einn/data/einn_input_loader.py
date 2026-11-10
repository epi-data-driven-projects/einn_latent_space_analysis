import os
from typing import Tuple

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
    Data orchestration class responsible for generating, calibrating, scaling,
    and windowing datasets before handing them off to the EINN Runner.
    """

    def __init__(
            self,
            true_model: BaseODEModel,
            calibrator_model: BaseODEModel,
            target_idx: int,
            device: str = 'cpu',
            cache_dir: str = "./cache"
    ):
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
            calibrator_epoch: int = 100,
            window_step: int = 20,
            window_min_days: int = 20
    ) -> Tuple[ExpandingWindowLoader, DataScalingStats]:
        """
        Generates synthetic states, pre-calibrates auxiliary data, normalizes them,
        and prepares the ExpandingWindowLoader for training.
        """
        cache_file = os.path.join(self.cache_dir, f"{num_states}_states_target_{self.target_idx}.pt")
        calibrator = ODESolutionGenerator(assumed_ode_model=self.calibrator_model, d_p=d_p, device=self.device)

        x_train, y_train, aux_train = [], [], []
        generate_data = True

        # Load from cache if available
        if os.path.exists(cache_file):
            print(f"Found cache: {cache_file}")
            cached_data = torch.load(cache_file, weights_only=False)
            if len(cached_data["x_train"]) >= num_states:
                x_train = cached_data["x_train"][:num_states]
                y_train = cached_data["y_train"][:num_states]
                aux_train = cached_data["aux_train"][:num_states]
                generate_data = False
                print("Data loaded from cache.")

        # Generate and Pre-calibrate if not cached
        if generate_data:
            print(f"Generating and pre-calibrating {num_states} diverse states...")
            for i in range(num_states):
                # Randomize physical dynamics
                true_params = [
                    np.random.uniform(0.3, 0.6),  # beta
                    np.random.uniform(0.1, 0.3),  # alpha
                    np.random.uniform(0.05, 0.15),  # gamma
                    np.random.uniform(0.01, 0.08)  # mu
                ]
                correlations = {
                    'mobility': np.random.uniform(-0.9, -0.4),
                    'temperature': np.random.uniform(-0.4, 0.1),
                    'testing': np.random.uniform(0.5, 0.9),
                    'stringency': np.random.uniform(0.7, 0.95)
                }

                generator = SyntheticDataGenerator(true_ode_model=self.true_model, seq_len=seq_len)
                data = generator.generate(
                    initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
                    true_params=true_params, noise_std=0.02, correlations=correlations
                )

                aux = calibrator.calibrate(
                    y_raw=data["y_raw"], initial_states=[0.99, 0.01, 0.0, 0.0, 0.0], epochs=calibrator_epoch, lr=0.001
                )

                # Anchor targeted compartment
                aux[:, self.target_idx] = data["y_raw"].ravel()

                x_train.append(data["x_raw"])
                y_train.append(data["y_raw"])
                aux_train.append(aux)

            torch.save({"x_train": x_train, "y_train": y_train, "aux_train": aux_train}, cache_file)

        # Z-Score Scaling
        preprocessor = DataPreprocessor()
        scaled_dict, scaling_stats = preprocessor.fit_transform(
            x_raw=np.concatenate(x_train, axis=0),
            y_raw=np.concatenate(y_train, axis=0),
            aux_raw=np.concatenate(aux_train, axis=0),
            target_idx=self.target_idx,
            initial_states=[0.99, 0.01, 0.0, 0.0, 0.0]
        )

        # Transform lists using the fitted scalers
        t_unscaled = np.arange(seq_len).reshape(-1, 1)
        x_scaled, y_scaled, aux_scaled, t_list = [], [], [], []

        for i in range(num_states):
            x_scaled.append(preprocessor.x_scaler.transform(x_train[i]))
            y_scaled.append(preprocessor.y_scaler.transform(y_train[i]))
            aux_scaled.append(preprocessor.aux_scaler.transform(aux_train[i]))
            t_list.append(t_unscaled)

        dataset = ExpandingWindowLoader(
            x_list=x_scaled, y_list=y_scaled, t_list=t_list, aux_list=aux_scaled,
            min_days=window_min_days, max_days=seq_len, step=window_step, device=self.device
        )

        return dataset, scaling_stats
