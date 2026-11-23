from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from einn.data.data_preprocessor import DataPreprocessor
from einn.data.data_scaling_stats import DataScalingStats
from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn_multimodel.data.expanding_window_loader import ExpandingWindowLoader
from einn_multimodel.model.model_labels import validate_model_registry
from einn.ode.sir_model import SIRModel
from einn.ode.seir_model import SEIRModel
from einn_multimodel.ode.sis_model import SISModel
from einn_multimodel.ode.sir_seir_compatible_model import SIRSEIRCompatibleModel


DEFAULT_MODEL_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "SIR": {"beta": (0.3, 0.6), "gamma": (0.05, 0.2)},
    "SIS": {"beta": (0.3, 0.6), "gamma": (0.05, 0.2)},
    "SEIR": {
        "beta": (0.3, 0.6),
        "alpha": (0.1, 0.5),
        "gamma": (0.05, 0.2),
    },
}

PARAMETER_ORDER = {
    "SIR": ("beta", "gamma"),
    "SIS": ("beta", "gamma"),
    "SEIR": ("beta", "alpha", "gamma"),
}

SUPPORTED_MODEL_PAIRS = {
    frozenset(("SIR", "SIS")),
    frozenset(("SIR", "SEIR")),
}


class MultiModelInputLoader:
    """Generate synthetic trajectories, optionally with every model assumption.

    pair_with_all_assumptions=False generates only correctly labelled examples.
    The assumed_model_label key then remains only as a compatibility alias.
    """

    def __init__(
        self,
        device: str = "cpu",
        population_n: float = 1.0,
        seed: int = 42,
    ):
        """Initialize MultiModelInputLoader to generate and standardize labeled synthetic trajectories across mechanistic models.


        :param str device: PyTorch device used for computations.
        :param float population_n: Total modeled population, or one for population-fraction states.
        :param int seed: Seed used to control pseudorandom number generation.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.device = device
        self.population_n = float(population_n)
        self.rng = np.random.default_rng(seed)
        self.preprocessor: Optional[DataPreprocessor] = None

    @staticmethod
    def _resolve_layout(model_names: tuple[str, ...]):
        """Perform resolve layout as part of generate and standardize labeled synthetic trajectories across mechanistic models.


        :param tuple[str, ...] model_names: Ordered epidemiological model identifiers.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
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

    def _make_model(self, name: str, model_names: tuple[str, ...]):
        """Perform make model as part of generate and standardize labeled synthetic trajectories across mechanistic models.


        :param str name: Identifier used for selecting the corresponding model or saved artifact.
        :param tuple[str, ...] model_names: Ordered epidemiological model identifiers.
        :return Any: The computed objects, metrics, or artifact paths for the requested operation.
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
        """Perform resolve param ranges as part of generate and standardize labeled synthetic trajectories across mechanistic models.


        :param tuple[str, ...] model_names: Ordered epidemiological model identifiers.
        :param Optional[Dict[str, Tuple[float, float]]] param_ranges: Allowed intervals for sampled mechanistic parameters.
        :param Optional[Dict[str, Dict[str, Tuple[float, float]]]] model_param_ranges: Sampling intervals for physical parameters of each model family.
        :return dict[str, dict[str, tuple[float, float]]]: Named tensors, metrics, or model components produced by this operation.
        """
        resolved = deepcopy(DEFAULT_MODEL_PARAM_RANGES)

        # Backward compatibility: a legacy shared range dictionary overrides
        # matching parameter names for every selected model.
        if param_ranges is not None:
            for name in model_names:
                for key, bounds in param_ranges.items():
                    if key in resolved[name]:
                        resolved[name][key] = tuple(bounds)

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
        """Perform sample params as part of generate and standardize labeled synthetic trajectories across mechanistic models.


        :param str model_name: Name of the mechanistic epidemic model, such as SIR, SIS, or ROST.
        :param dict[str, dict[str, tuple[float, float]]] ranges: Model-specific ranges used to sample mechanistic parameters.
        :return tuple[dict[str, float], list[float]]: Computed result in the configured output representation.
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
        **_kwargs,
    ) -> tuple[ExpandingWindowLoader, DataScalingStats]:
        """Construct prepare training data using the configured model and experiment settings.


        :param int num_states_per_model: Number of independent synthetic trajectories per epidemiological model.
        :param int seq_len: Number of time coordinates per generated trajectory.
        :param Optional[list[float]] initial_states: Initial compartment populations in model state order.
        :param Optional[Dict[str, Tuple[float, float]]] param_ranges: Allowed intervals for sampled mechanistic parameters.
        :param Optional[Dict[str, Dict[str, Tuple[float, float]]]] model_param_ranges: Sampling intervals for physical parameters of each model family.
        :param Optional[Dict[str, Tuple[float, float]]] correlation_ranges: Sampling intervals for the covariate correlation coefficients.
        :param float noise_std: Relative observation-noise scale configured for the experiment.
        :param int window_step: Interval between successive expanding observation-window lengths.
        :param int window_min_days: Minimum number of observed time points in each training prefix.
        :param bool randomize_records: Whether to permute the order of generated trajectory records.
        :param tuple[str, ...] model_names: Ordered epidemiological model identifiers.
        :param bool fit_scalers: Whether to fit new normalization statistics on this dataset.
        :param Optional[DataScalingStats] scaling_stats: Statistics fitted on training data for physical-domain conversion.
        :param bool pair_with_all_assumptions: Whether each trajectory is paired with all candidate assumed model labels.
        :param dict _kwargs: Additional named arguments.
        :return tuple[ExpandingWindowLoader, DataScalingStats]: Computed result in the configured output representation.
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

        records: list[dict] = []
        print(
            f"Generating {num_states_per_model} independent trajectories per model; "
            f"selected models={model_names}; "
            f"routing={'all assumptions' if pair_with_all_assumptions else 'TRUE MODEL ONLY'}."
        )

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

                # In the true-only experiment no cross-assumption copies are generated.
                for assumed_name in (model_names if pair_with_all_assumptions else (true_name,)):
                    if assumed_name == true_name:
                        aux = generated["true_latent_states"].copy()
                        assumed_param_dict = true_param_dict
                    else:
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

                    # Anchor the assumed-model trajectory to the observed,
                    # clean infected curve. The noisy observation remains in x.
                    aux[:, infection_idx] = generated["y_raw"].ravel()

                    records.append(
                        {
                            "x": generated["x_raw"],
                            "y": generated["y_raw"],
                            "aux": aux,
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

        x_all = np.concatenate([record["x"] for record in records], axis=0)
        y_all = np.concatenate([record["y"] for record in records], axis=0)
        aux_all = np.concatenate([record["aux"] for record in records], axis=0)

        if fit_scalers:
            # Fit normalization statistics exclusively on the training trajectories.
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

            scaling_stats.t_min = torch.tensor(0.0, dtype=torch.float32)
            scaling_stats.t_max = torch.tensor(float(seq_len - 1), dtype=torch.float32)
        else:
            # Transform held-out trajectories with the already-fitted training scalers.
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

        x_scaled = scaled["x"].cpu().numpy()
        y_scaled = scaled["y"].cpu().numpy()
        aux_scaled = scaled["aux_targets"].cpu().numpy()
        t_local = np.arange(seq_len, dtype=np.float32).reshape(-1, 1)

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
