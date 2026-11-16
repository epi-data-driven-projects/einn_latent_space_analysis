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
            target_idx: int,
            device: str = 'cpu',
            cache_dir: str = "./cache"
    ):
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
            **kwargs  # Felfogja a felesleges calibrator argumentumokat, ha hívásnál megadod őket
    ) -> Tuple[ExpandingWindowLoader, DataScalingStats]:

        d_s = self.true_model.d_s
        if initial_states is None:
            initial_states = [0.99, 0.01] + [0.0] * (d_s - 2)

        if correlation_ranges is None:
            correlation_ranges = {
                'mobility': (-0.9, -0.4), 'temperature': (-0.4, 0.1),
                'testing': (0.5, 0.9), 'stringency': (0.7, 0.95)
            }

        # A seq_len is bekerül a fájlnévbe, hogy ne töltsön be rossz hosszúságú adatot
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

                # 1. KEZDETI ÁLLAPOTOK (A járvány berúgása)
                current_initial_states = initial_states.copy()
                if d_s == 18:  # Kifejezetten a Röst modell (18 rekesz) esetén
                    current_initial_states = [0.0] * d_s
                    current_initial_states[0] = 0.99
                    current_initial_states[3] = 0.01  # Ip (Pre-szimptomatikus fertőző) rekeszbe tesszük a betegeket

                # 2. PARAMÉTEREK SORSOLÁSA (Nincs több beégetett / fix érték!)
                if param_ranges is not None and len(param_ranges) == d_p:
                    true_params = [np.random.uniform(*v) for v in param_ranges.values()]
                else:
                    # Biztonsági fallback (ha elfelejtenél range-t átadni)
                    true_params = [np.random.uniform(0.05, 0.4) for _ in range(d_p)]
                    true_params[0] = np.random.uniform(0.4, 0.7)  # Magas beta a kitöréshez

                # 3. KORRELÁCIÓK SORSOLÁSA
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

                # Lekérjük a tiszta fizikai megoldást (.copy() fontos, hogy memóriában elkülönüljön)
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
