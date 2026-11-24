"""
Multi-model synthetic trajectory generator, preprocessor, and ExpandingWindowLoader builder.

Supports the SIR-SIS and SIR-SEIR model pairs studied in thesis Sections 2.4 and 4.2,
unifying compartment dimensions across paired models (including zero-padding the E
compartment for SIR in SIR-SEIR runs) and enforcing strict train/test scaler separation
(Section 3).
"""
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from einn.data.data_preprocessor import DataPreprocessor
from einn.data.data_scaling_stats import DataScalingStats
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.seir_model import SEIRModel
from einn.ode.sir_model import SIRModel
from einn_multimodel.data.expanding_window_loader import ExpandingWindowLoader
from einn_multimodel.model.model_labels import validate_model_registry
from einn_multimodel.ode.sir_seir_compatible_model import SIRSEIRCompatibleModel
from einn_multimodel.ode.sis_model import SISModel

# Default uniform sampling intervals for mechanistic ODE parameters per model family.
DEFAULT_MODEL_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "SIR": {"beta": (0.3, 0.6), "gamma": (0.05, 0.2)},
    "SIS": {"beta": (0.3, 0.6), "gamma": (0.05, 0.2)},
    "SEIR": {
        "beta": (0.3, 0.6),
        "alpha": (0.1, 0.5),
        "gamma": (0.05, 0.2),
    },
}

# Canonical parameter ordering expected by each ODE model's forward signature.
PARAMETER_ORDER: dict[str, tuple[str, ...]] = {
    "SIR": ("beta", "gamma"),
    "SIS": ("beta", "gamma"),
    "SEIR": ("beta", "alpha", "gamma"),
}

# Epidemiological model pairs supported by the unified OutputModule layout (Section 2.4).
SUPPORTED_MODEL_PAIRS: set[frozenset[str]] = {
    frozenset(("SIR", "SIS")),
    frozenset(("SIR", "SEIR")),
}


class MultiModelInputLoader:
    """
    Generate, standardize, and package labeled synthetic trajectories for two-model EINN runs.

    When pair_with_all_assumptions is True, each simulated trajectory is paired with every
    candidate assumed model label (supporting matched/mismatched control experiments in
    thesis Section 5). When False, only true-label records are created and assumed_model_label
    acts as a compatibility alias for true_model_label.
    """

    def __init__(
        self,
        device: str = "cpu",
        population_n: float = 1.0,
        seed: int = 42,
    ) -> None:
        """
        Initialize the multi-model loader with compute device, population scale, and RNG seed.

        :param str device: Target PyTorch device string for the output ExpandingWindowLoader.
        :param float population_n: Total population N (1.0 for normalized population fractions).
        :param int seed: Random seed for reproducible parameter, noise, and covariate sampling.
        :return None: Initializes internal RNG and empty preprocessor reference in place.
        """
        self.device: str = device
        self.population_n: float = float(population_n)
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.preprocessor: Optional[DataPreprocessor] = None

    @staticmethod
    def _resolve_layout(model_names: tuple[str, ...]) -> dict[str, Any]:
        """
        Resolve the shared state dimension d_s, target index, and default initial states for a model pair.

        For SIR-SIS, uses 3 compartments (S, I, R/pad) with infection_idx = 1.
        For SIR-SEIR, uses 4 compartments (S, E, I, R) with infection_idx = 2 (Section 2.4).

        :param tuple[str, ...] model_names: Two uppercase epidemiological model identifiers.
        :return dict[str, Any]: Layout dict with keys d_s (int), infection_idx (int),
            and default_initial_states (list[float] of length d_s).
        :raises ValueError: If model_names is not one of the supported pairs.
        """
        pair = frozenset(model_names)
        if pair == frozenset(("SIR", "SIS")):
            return {
                "d_s": 3,
                "infection_idx": 1,
                "default_initial_states": [0.99, 0.01, 0.0],
            }
        if pair == frozenset(("SIR", "SEIR")):
            return {
                "d_s": 4,
                "infection_idx": 2,
                "default_initial_states": [0.99, 0.0, 0.01, 0.0],
            }
        raise ValueError(
            "Supported model pairs are ('SIR', 'SIS') and ('SIR', 'SEIR')."
        )

    def _make_model(
        self,
        name: str,
        model_names: tuple[str, ...],
    ) -> SIRModel | SISModel | SEIRModel | SIRSEIRCompatibleModel:
        """
        Instantiate the ODE model for name with compartment layout compatible with model_names.

        When SIR is paired with SEIR, returns SIRSEIRCompatibleModel so SIR outputs a 4-state
        vector with a constant-zero E compartment (thesis Section 2.4).

        :param str name: Uppercase model identifier (SIR, SIS, or SEIR).
        :param tuple[str, ...] model_names: Active two-model pair determining the shared layout.
        :return SIRModel | SISModel | SEIRModel | SIRSEIRCompatibleModel: Configured ODE model instance.
        :raises ValueError: If name is not a supported model.
        """
        pair = frozenset(model_names)

        if name == "SIR" and pair == frozenset(("SIR", "SEIR")):
            return SIRSEIRCompatibleModel(
                population_n=self.population_n,
                target_idx=2,
            )
        if name == "SIR":
            return SIRModel(
                population_n=self.population_n,
                target_idx=1,
            )
        if name == "SIS":
            return SISModel(
                population_n=self.population_n,
                target_idx=1,
            )
        if name == "SEIR":
            return SEIRModel(
                population_n=self.population_n,
                target_idx=2,
            )
        raise ValueError(f"Unsupported model: {name}")

    def _resolve_param_ranges(
        self,
        model_names: tuple[str, ...],
        param_ranges: Optional[Dict[str, Tuple[float, float]]],
        model_param_ranges: Optional[
            Dict[str, Dict[str, Tuple[float, float]]]
        ],
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """
        Merge default, legacy shared, and per-model ODE parameter sampling intervals and validate them.

        :param tuple[str, ...] model_names: Active two-model pair.
        :param Optional[Dict[str, Tuple[float, float]]] param_ranges: Optional legacy shared
            parameter bounds overriding matching parameter names across all selected models.
        :param Optional[Dict[str, Dict[str, Tuple[float, float]]]] model_param_ranges: Optional
            model-specific parameter bounds mapping model_name -> param_name -> (low, high).
        :return dict[str, dict[str, tuple[float, float]]]: Validated per-model parameter intervals.
        :raises ValueError: If unknown models/parameters are supplied, required parameters are
            missing, or any interval does not satisfy finite low < high.
        """
        resolved = deepcopy(DEFAULT_MODEL_PARAM_RANGES)

        # Apply legacy shared parameter bounds to any selected model that defines that parameter.
        if param_ranges is not None:
            for name in model_names:
                for key, bounds in param_ranges.items():
                    if key in resolved[name]:
                        resolved[name][key] = tuple(bounds)

        # Apply explicit model-specific overrides after verifying model and parameter names.
        if model_param_ranges is not None:
            unknown_models = set(model_param_ranges).difference(model_names)
            if unknown_models:
                raise ValueError(
                    f"Parameter ranges supplied for unselected models: "
                    f"{sorted(unknown_models)}"
                )
            for name, ranges in model_param_ranges.items():
                unknown_params = set(ranges).difference(PARAMETER_ORDER[name])
                if unknown_params:
                    raise ValueError(
                        f"Unknown parameters for {name}: {sorted(unknown_params)}"
                    )
                resolved[name].update(
                    {key: tuple(value) for key, value in ranges.items()}
                )

        # Verify completeness and strict validity (low < high) for every active model parameter.
        for name in model_names:
            missing = set(PARAMETER_ORDER[name]).difference(resolved[name])
            if missing:
                raise ValueError(
                    f"Missing parameter ranges for {name}: {sorted(missing)}"
                )
            for key in PARAMETER_ORDER[name]:
                low, high = resolved[name][key]
                if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                    raise ValueError(
                        f"Invalid range for {name}.{key}: {(low, high)}"
                    )
        return resolved

    def _sample_params(
        self,
        model_name: str,
        ranges: dict[str, dict[str, tuple[float, float]]],
    ) -> tuple[dict[str, float], list[float]]:
        """
        Draw one uniform random ODE parameter vector for model_name in canonical PARAMETER_ORDER.

        :param str model_name: Uppercase model identifier (SIR, SIS, or SEIR).
        :param dict[str, dict[str, tuple[float, float]]] ranges: Resolved sampling intervals per model.
        :return tuple[dict[str, float], list[float]]: Pair (param_dict, param_list) containing
            named parameter values and their ordered list of length d_p.
        """
        values = {
            key: float(self.rng.uniform(*ranges[model_name][key]))
            for key in PARAMETER_ORDER[model_name]
        }
        return values, [values[key] for key in PARAMETER_ORDER[model_name]]

    def prepare_training_data(
        self,
        num_states_per_model: int = 50,
        seq_len: int = 200,
        initial_states: Optional[list[float]] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        model_param_ranges: Optional[
            Dict[str, Dict[str, Tuple[float, float]]]
        ] = None,
        correlation_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        noise_std: float = 0.02,
        window_step: int = 20,
        window_min_days: int = 20,
        randomize_records: bool = True,
        model_names: tuple[str, ...] = ("SIR", "SIS"),
        fit_scalers: bool = True,
        scaling_stats: Optional[DataScalingStats] = None,
        pair_with_all_assumptions: bool = True,
        **_kwargs: Any,
    ) -> tuple[ExpandingWindowLoader, DataScalingStats]:
        """
        Generate synthetic multi-model trajectories, standardize them, and build the window loader.

        Each generated record has arrays x of shape (seq_len, 5), y of shape (seq_len, 1), and
        aux of shape (seq_len, d_s). When fit_scalers is True, Z-score scalers are fitted once
        on the concatenated training set of shape (N_records * seq_len, *); when False, the
        previously fitted training scalers are applied to prevent test-set leakage (Section 3).

        :param int num_states_per_model: Number of independent trajectories simulated per model family.
        :param int seq_len: Number of timesteps T per simulated trajectory.
        :param Optional[list[float]] initial_states: Physical initial compartment proportions s(0)
            of length d_s; uses layout defaults if None.
        :param Optional[Dict[str, Tuple[float, float]]] param_ranges: Optional legacy shared parameter bounds.
        :param Optional[Dict[str, Dict[str, Tuple[float, float]]]] model_param_ranges: Optional
            per-model parameter sampling bounds.
        :param Optional[Dict[str, Tuple[float, float]]] correlation_ranges: Target-correlation bounds
            for the 4 synthetic covariates (mobility, temperature, testing, stringency).
        :param float noise_std: Relative Gaussian observation-noise scale applied to the target.
        :param int window_step: Step size between consecutive expanding-window prefix lengths.
        :param int window_min_days: Shortest observation prefix length included in the dataset.
        :param bool randomize_records: If True, randomly permutes the generated records before slicing.
        :param tuple[str, ...] model_names: Two distinct model names forming a supported pair.
        :param bool fit_scalers: If True, fits new training scalers; if False, reuses self.preprocessor.
        :param Optional[DataScalingStats] scaling_stats: Training scaling statistics required when
            fit_scalers is False.
        :param bool pair_with_all_assumptions: If True, pairs each trajectory with every model in
            model_names as assumed_model_label; if False, only creates matched true-label records.
        :param Any _kwargs: Ignored extra keyword arguments retained for caller compatibility.
        :return tuple[ExpandingWindowLoader, DataScalingStats]: Pair (dataset, scaling_stats)
            containing the standardized multi-model expanding-window dataset and scaling metadata.
        :raises ValueError: If trajectory counts, horizon, model pair, initial states, or
            fit_scalers=False arguments are invalid.
        :raises TypeError: If pair_with_all_assumptions is not a bool.
        :raises RuntimeError: If fit_scalers is False before training scalers have been fitted.
        """
        if num_states_per_model < 1:
            raise ValueError("num_states_per_model must be at least 1.")
        if seq_len < window_min_days:
            raise ValueError("seq_len must be >= window_min_days.")

        model_names = tuple(name.upper() for name in model_names)
        if len(model_names) != 2 or len(set(model_names)) != 2:
            raise ValueError("Exactly two distinct model names are required.")

        model_to_id = validate_model_registry(list(model_names))
        layout = self._resolve_layout(model_names)
        infection_idx = layout["infection_idx"]

        if initial_states is None:
            initial_states = layout["default_initial_states"]
        if len(initial_states) != layout["d_s"]:
            raise ValueError(
                f"Expected {layout['d_s']} initial states for {model_names}, "
                f"got {len(initial_states)}."
            )

        if correlation_ranges is None:
            correlation_ranges = {
                "mobility": (-0.9, -0.4),
                "temperature": (-0.4, 0.1),
                "testing": (0.5, 0.9),
                "stringency": (0.7, 0.95),
            }

        ranges = self._resolve_param_ranges(
            model_names=model_names,
            param_ranges=param_ranges,
            model_param_ranges=model_param_ranges,
        )

        if not isinstance(pair_with_all_assumptions, bool):
            raise TypeError("pair_with_all_assumptions must be bool.")

        records: list[dict[str, Any]] = []
        print(
            f"Generating {num_states_per_model} independent trajectories per model; "
            f"selected models={model_names}; "
            f"routing={'all assumptions' if pair_with_all_assumptions else 'TRUE MODEL ONLY'}."
        )

        # Simulate num_states_per_model trajectories per true model family.
        for true_name in model_names:
            for state_idx in range(num_states_per_model):
                true_param_dict, true_param_list = self._sample_params(
                    true_name, ranges
                )
                correlations = {
                    key: float(self.rng.uniform(*bounds))
                    for key, bounds in correlation_ranges.items()
                }

                true_model = self._make_model(true_name, model_names)
                true_generator = SyntheticDataGenerator(
                    true_ode_model=true_model,
                    seq_len=seq_len,
                    rng=self.rng,
                )
                generated = true_generator.generate(
                    initial_states=initial_states,
                    true_params=true_param_list,
                    noise_std=noise_std,
                    correlations=correlations,
                )

                # Create either matched-only records or full cross-assumption pairs.
                for assumed_name in (model_names if pair_with_all_assumptions else (true_name,)):
                    if assumed_name == true_name:
                        aux = generated["true_latent_states"].copy()
                        assumed_param_dict = true_param_dict
                    else:
                        # Simulate an auxiliary reference trajectory under the mismatched assumed ODE.
                        assumed_param_dict, assumed_param_list = self._sample_params(
                            assumed_name, ranges
                        )
                        assumed_model = self._make_model(
                            assumed_name, model_names
                        )
                        assumed_generator = SyntheticDataGenerator(
                            true_ode_model=assumed_model,
                            seq_len=seq_len,
                            rng=self.rng,
                        )
                        aux = assumed_generator.generate_trajectory(
                            initial_states=initial_states,
                            true_params=assumed_param_list,
                        )

                    # Anchor the target compartment in aux to the clean ground-truth curve; x keeps noisy observations.
                    aux[:, infection_idx] = generated["y_raw"].ravel()

                    records.append(
                        {
                            "x": generated["x_raw"],            # Shape: (seq_len, 5)
                            "y": generated["y_raw"],            # Shape: (seq_len, 1)
                            "aux": aux,                         # Shape: (seq_len, d_s)
                            "true_label": model_to_id[true_name],
                            "assumed_label": model_to_id[assumed_name],
                            "true_name": true_name,
                            "assumed_name": assumed_name,
                            "true_params": true_param_dict,
                            "assumed_params": assumed_param_dict,
                            "trajectory_id": f"{true_name}_{state_idx}",
                        }
                    )

        if randomize_records:
            permutation = self.rng.permutation(len(records))
            records = [records[int(i)] for i in permutation]

        # Concatenate all records along the time axis for vectorized standardization: (N_records * seq_len, *).
        x_all = np.concatenate([record["x"] for record in records], axis=0)
        y_all = np.concatenate([record["y"] for record in records], axis=0)
        aux_all = np.concatenate([record["aux"] for record in records], axis=0)

        if fit_scalers:
            # Fit Z-score scalers exclusively on training data (Section 3).
            preprocessor = DataPreprocessor()
            scaled, scaling_stats = preprocessor.fit_transform(
                x_raw=x_all,
                y_raw=y_all,
                aux_raw=aux_all,
                target_idx=infection_idx,
                initial_states=initial_states,
                time_length=seq_len,
            )
            self.preprocessor = preprocessor

            # Fix temporal min-max bounds to a single trajectory horizon [0, seq_len - 1].
            scaling_stats.t_min = torch.tensor(0.0, dtype=torch.float32)
            scaling_stats.t_max = torch.tensor(float(seq_len - 1), dtype=torch.float32)
        else:
            # Apply previously fitted training scalers to held-out evaluation trajectories.
            if self.preprocessor is None:
                raise RuntimeError(
                    "Cannot prepare test data before fitting the training scalers. "
                    "Call prepare_training_data(..., fit_scalers=True) first."
                )
            if scaling_stats is None:
                raise ValueError(
                    "The training scaling_stats must be supplied when fit_scalers=False."
                )

            aux_all = aux_all.copy()
            aux_all[:, infection_idx] = y_all.ravel()
            scaled = {
                "x": torch.as_tensor(
                    self.preprocessor.x_scaler.transform(x_all),
                    dtype=torch.float32,
                ),
                "y": torch.as_tensor(
                    self.preprocessor.y_scaler.transform(y_all),
                    dtype=torch.float32,
                ),
                "aux_targets": torch.as_tensor(
                    self.preprocessor.aux_scaler.transform(aux_all),
                    dtype=torch.float32,
                ),
            }

        # Slice standardized arrays back into per-trajectory lists of shape (seq_len, *).
        x_scaled = scaled["x"].cpu().numpy()
        y_scaled = scaled["y"].cpu().numpy()
        aux_scaled = scaled["aux_targets"].cpu().numpy()
        t_local = np.arange(seq_len, dtype=np.float32).reshape(-1, 1)  # Shape: (seq_len, 1)

        x_list, y_list, aux_list, t_list = [], [], [], []
        true_labels, assumed_labels = [], []
        for idx, record in enumerate(records):
            start = idx * seq_len
            end = (idx + 1) * seq_len
            x_list.append(x_scaled[start:end])
            y_list.append(y_scaled[start:end])
            aux_list.append(aux_scaled[start:end])
            t_list.append(t_local.copy())
            true_labels.append(record["true_label"])
            assumed_labels.append(record["assumed_label"])

        dataset = ExpandingWindowLoader(
            x_list=x_list,
            y_list=y_list,
            t_list=t_list,
            aux_list=aux_list,
            min_days=window_min_days,
            max_days=seq_len,
            step=window_step,
            device=self.device,
            true_model_labels=true_labels,
            assumed_model_labels=assumed_labels,
        )
        return dataset, scaling_stats
