
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
    def __init__(
            self,
            true_model: BaseODEModel,
            calibrator_model: BaseODEModel, # Preserved compatibility argument; not used by the current synthetic data path.
            target_idx: int,
            device: str = 'cpu',
            cache_dir: str = "./cache"
    ):
        """Initialize EINNInputLoader to store the configuration and initialize internal state.


        :param BaseODEModel true_model: Mechanistic model used to generate reference data.
        :param BaseODEModel calibrator_model: Mechanistic model instance retained for compatibility with the data-loader API.
        :param int target_idx: Index of the epidemiological target compartment.
        :param str device: PyTorch device used for computations.
        :param str cache_dir: Directory containing cached synthetic trajectories and metadata.
        :return None: No return value; state or serialized artifacts are updated in place.
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
            **kwargs # Accept legacy optional loader keywords without changing data generation.
    ) -> Tuple[ExpandingWindowLoader, DataScalingStats]:

        """Construct prepare training data using the configured model and experiment settings.


        :param int num_states: Number of independent synthetic trajectories to generate.
        :param int seq_len: Number of time coordinates per generated trajectory.
        :param int d_p: Number of trainable or sampled mechanistic parameters.
        :param Optional[List[float]] initial_states: Initial compartment populations in model state order.
        :param Optional[Dict[str, Tuple[float, float]]] param_ranges: Allowed intervals for sampled mechanistic parameters.
        :param Optional[Dict[str, Tuple[float, float]]] correlation_ranges: Sampling intervals for the covariate correlation coefficients.
        :param float noise_std: Relative observation-noise scale configured for the experiment.
        :param int window_step: Interval between successive expanding observation-window lengths.
        :param int window_min_days: Minimum number of observed time points in each training prefix.
        :param dict kwargs: Additional named arguments.
        :return Tuple[ExpandingWindowLoader, DataScalingStats]: Computed result in the configured output representation.
        """
        d_s = self.true_model.d_s
        if initial_states is None:
            initial_states = [0.99, 0.01] + [0.0] * (d_s - 2)

        if correlation_ranges is None:
            correlation_ranges = {
                'mobility': (-0.9, -0.4), 'temperature': (-0.4, 0.1),
                'testing': (0.5, 0.9), 'stringency': (0.7, 0.95)
            }

        # Use sequence length in the cache filename to reject different time horizons.
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

                # Initial conditions for the synthetic outbreak.
                current_initial_states = initial_states.copy()
                if d_s == 18:  # Use the original 18-state Röst initialization policy for synthetic outbreaks.
                    current_initial_states = [0.0] * d_s
                    current_initial_states[0] = 0.99
                    current_initial_states[3] = 0.01  # Seed the presymptomatic compartment at one percent of the population.

                # Sample physical parameters from the configured ranges.
                if param_ranges is not None and len(param_ranges) == d_p:
                    true_params = [np.random.uniform(*v) for v in param_ranges.values()]
                else:
                    # Fallback parameter ranges if none are provided.
                    true_params = [np.random.uniform(0.05, 0.4) for _ in range(d_p)]
                    true_params[0] = np.random.uniform(0.4, 0.7) # Fallback increases the transmission rate to create a visible outbreak.

                # Sample correlations for synthetic external variables.
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

                # Copy clean numerical reference states; do not alias the original array.
                aux = data["true_latent_states"].copy()

                aux[:, self.target_idx] = data["y_raw"].ravel()

                x_train.append(data["x_raw"])
                y_train.append(data["y_raw"])
                aux_train.append(aux)

            torch.save({"x_train": x_train, "y_train": y_train, "aux_train": aux_train}, cache_file)
            print("Synthetic dataset successfully cached.")

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
