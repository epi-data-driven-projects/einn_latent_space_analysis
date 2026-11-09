import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from einn.data.ode_solution_generator import ODESolutionGenerator

from einn.data.synthetic_data_generator import SyntheticDataGenerator
from einn.ode.base_ode_model import BaseODEModel


@dataclass
class GeneratedSeries:
    """One synthetic trajectory and its assumed-model auxiliary solution."""

    x_raw: np.ndarray
    y_raw: np.ndarray
    aux_raw: np.ndarray
    true_latent_states: np.ndarray
    true_params: dict[str, float]
    correlations: dict[str, float]


class InputLoader:
    """Generate/load synthetic trajectories and cache their calibrated aux data.

    The cache stores raw X/y, true latent states, sampled parameters, correlations,
    and the ODE-precalibrated auxiliary trajectories. If the generation settings
    change, the cache metadata no longer matches and the data are regenerated.
    """

    DEFAULT_PARAMETER_RANGES = {
        "beta": (0.30, 0.60),
        "alpha": (0.10, 0.30),
        "gamma": (0.05, 0.15),
        "mu": (0.01, 0.08),
    }
    DEFAULT_CORRELATION_RANGES = {
        "mobility": (-0.90, -0.40),
        "temperature": (-0.40, 0.10),
        "testing": (0.50, 0.90),
        "stringency": (0.70, 0.95),
    }

    def __init__(
        self,
        true_model: BaseODEModel,
        assumed_model: BaseODEModel,
        n_series: int = 20,
        seq_len: int = 200,
        true_initial_states: Optional[list[float]] = None,
        assumed_initial_states: Optional[list[float]] = None,
        parameter_ranges: Optional[dict[str, tuple[float, float]]] = None,
        parameter_names: Optional[list[str]] = None,
        correlations: Optional[dict[str, float]] = None,
        correlation_ranges: Optional[dict[str, tuple[float, float]]] = None,
        noise_std: float = 0.02,
        calibration_epochs: int = 100,
        calibration_lr: float = 0.01,
        calibration_initial_guess: Optional[list[float]] = None,
        cache_path: str | Path = "./data_cache/einn_generated_data.pt",
        seed: int = 42,
        calibration_device: str = "cpu",
    ) -> None:
        if n_series < 1:
            raise ValueError("n_series must be at least 1.")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if true_model.d_s != len(true_initial_states or self._default_initial_states(true_model.d_s)):
            raise ValueError("true_initial_states length must match true_model.d_s.")
        if assumed_model.d_s != len(
            assumed_initial_states or self._default_initial_states(assumed_model.d_s)
        ):
            raise ValueError("assumed_initial_states length must match assumed_model.d_s.")

        self.true_model = true_model
        self.assumed_model = assumed_model
        self.n_series = n_series
        self.seq_len = seq_len
        self.true_initial_states = list(
            true_initial_states or self._default_initial_states(true_model.d_s)
        )
        self.assumed_initial_states = list(
            assumed_initial_states or self._default_initial_states(assumed_model.d_s)
        )

        self.parameter_names = parameter_names or self._infer_parameter_names(true_model)
        self.parameter_ranges = parameter_ranges or {
            name: self.DEFAULT_PARAMETER_RANGES[name]
            for name in self.parameter_names
        }
        missing = set(self.parameter_names) - set(self.parameter_ranges)
        if missing:
            raise ValueError(f"Missing parameter ranges for: {sorted(missing)}")

        self.correlations = correlations
        self.correlation_ranges = correlation_ranges or self.DEFAULT_CORRELATION_RANGES
        self.noise_std = noise_std
        self.calibration_epochs = calibration_epochs
        self.calibration_lr = calibration_lr
        self.calibration_initial_guess = calibration_initial_guess
        self.cache_path = Path(cache_path)
        self.seed = seed
        self.calibration_device = calibration_device

        self.rng = np.random.default_rng(seed)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.calibrator = ODESolutionGenerator(
            assumed_ode_model=self.assumed_model,
            d_p=len(self._infer_parameter_names(self.assumed_model)),
            device=calibration_device,
        )

    @staticmethod
    def _default_initial_states(d_s: int) -> list[float]:
        defaults = {
            3: [0.99, 0.01, 0.0],
            5: [0.99, 0.01, 0.0, 0.0, 0.0],
        }
        if d_s not in defaults:
            raise ValueError(
                f"No default initial state is defined for d_s={d_s}; pass it explicitly."
            )
        return defaults[d_s]

    @staticmethod
    def _infer_parameter_names(model: BaseODEModel) -> list[str]:
        model_name = model.__class__.__name__.lower()
        if "sir" in model_name and "seirm" not in model_name:
            return ["beta", "gamma"]
        if "seirm" in model_name:
            return ["beta", "alpha", "gamma", "mu"]
        raise ValueError(
            "Could not infer ODE parameter order. Pass parameter_names explicitly."
        )

    def _sample_parameters(self) -> dict[str, float]:
        return {
            name: float(self.rng.uniform(*self.parameter_ranges[name]))
            for name in self.parameter_names
        }

    def _sample_correlations(self) -> dict[str, float]:
        if self.correlations is not None:
            return {key: float(value) for key, value in self.correlations.items()}
        return {
            name: float(self.rng.uniform(*bounds))
            for name, bounds in self.correlation_ranges.items()
        }

    def _default_calibration_guess(self) -> list[float]:
        names = self._infer_parameter_names(self.assumed_model)
        defaults = {"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05}
        return [defaults[name] for name in names]

    def _metadata(self) -> dict:
        return {
            "true_model": self.true_model.__class__.__name__,
            "true_d_s": self.true_model.d_s,
            "true_target_idx": self.true_model.target_idx,
            "assumed_model": self.assumed_model.__class__.__name__,
            "assumed_d_s": self.assumed_model.d_s,
            "assumed_target_idx": self.assumed_model.target_idx,
            "n_series": self.n_series,
            "seq_len": self.seq_len,
            "true_initial_states": self.true_initial_states,
            "assumed_initial_states": self.assumed_initial_states,
            "parameter_names": self.parameter_names,
            "parameter_ranges": self.parameter_ranges,
            "correlations": self.correlations,
            "correlation_ranges": self.correlation_ranges,
            "noise_std": self.noise_std,
            "calibration_epochs": self.calibration_epochs,
            "calibration_lr": self.calibration_lr,
            "calibration_initial_guess": self.calibration_initial_guess,
            "seed": self.seed,
        }

    def _generate_record(
        self,
        true_params: dict[str, float],
        correlations: dict[str, float],
        calibrate_aux: bool = True,
        noise_std: Optional[float] = None,
    ) -> GeneratedSeries:
        generator = SyntheticDataGenerator(
            true_ode_model=self.true_model,
            seq_len=self.seq_len,
        )
        generated = generator.generate(
            initial_states=self.true_initial_states,
            true_params=[true_params[name] for name in self.parameter_names],
            noise_std=self.noise_std if noise_std is None else noise_std,
            correlations=correlations,
        )

        if calibrate_aux:
            initial_guess = (
                self.calibration_initial_guess
                if self.calibration_initial_guess is not None
                else self._default_calibration_guess()
            )
            aux_raw = self.calibrator.calibrate(
                y_raw=generated["y_raw"],
                initial_states=self.assumed_initial_states,
                initial_guess=initial_guess,
                epochs=self.calibration_epochs,
                lr=self.calibration_lr,
            )
        else:
            aux_raw = np.full(
                (self.seq_len, self.assumed_model.d_s),
                np.nan,
                dtype=np.float32,
            )

        return GeneratedSeries(
            x_raw=np.asarray(generated["x_raw"], dtype=np.float32),
            y_raw=np.asarray(generated["y_raw"], dtype=np.float32),
            aux_raw=np.asarray(aux_raw, dtype=np.float32),
            true_latent_states=np.asarray(
                generated["true_latent_states"], dtype=np.float32
            ),
            true_params=true_params,
            correlations=correlations,
        )

    @staticmethod
    def _record_to_dict(record: GeneratedSeries) -> dict:
        return {
            "x_raw": record.x_raw,
            "y_raw": record.y_raw,
            "aux_raw": record.aux_raw,
            "true_latent_states": record.true_latent_states,
            "true_params": record.true_params,
            "correlations": record.correlations,
        }

    @staticmethod
    def _dict_to_record(data: dict) -> GeneratedSeries:
        return GeneratedSeries(
            x_raw=np.asarray(data["x_raw"], dtype=np.float32),
            y_raw=np.asarray(data["y_raw"], dtype=np.float32),
            aux_raw=np.asarray(data["aux_raw"], dtype=np.float32),
            true_latent_states=np.asarray(
                data["true_latent_states"], dtype=np.float32
            ),
            true_params={k: float(v) for k, v in data["true_params"].items()},
            correlations={k: float(v) for k, v in data["correlations"].items()},
        )

    def load_or_generate(self, force_regenerate: bool = False) -> list[GeneratedSeries]:
        metadata = self._metadata()
        if self.cache_path.exists() and not force_regenerate:
            try:
                try:
                    cached = torch.load(
                        self.cache_path, map_location="cpu", weights_only=False
                    )
                except TypeError:  # Older PyTorch versions
                    cached = torch.load(self.cache_path, map_location="cpu")

                if cached.get("metadata") == metadata:
                    self.logger.info("Loading generated data and aux trajectories from %s", self.cache_path)
                    return [self._dict_to_record(item) for item in cached["records"]]
                self.logger.info("Cache metadata differs from current settings; regenerating data.")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                self.logger.warning("Could not read cache (%s); regenerating it.", exc)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        records = []
        for index in range(self.n_series):
            params = self._sample_parameters()
            corr = self._sample_correlations()
            self.logger.info("Generating trajectory %d/%d", index + 1, self.n_series)
            records.append(self._generate_record(params, corr, calibrate_aux=True))

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": metadata,
                "records": [self._record_to_dict(record) for record in records],
            },
            self.cache_path,
        )
        self.logger.info("Generated trajectories and calibrated aux data saved to %s", self.cache_path)
        return records

    def generate_test_series(
        self,
        true_params: Optional[dict[str, float]] = None,
        correlations: Optional[dict[str, float]] = None,
        noise_std: Optional[float] = None,
    ) -> GeneratedSeries:
        """Generate one held-out trajectory; aux calibration is not needed for inference."""
        # Use a separate deterministic stream so that a cache hit does not
        # accidentally make the held-out series equal to the first training series.
        test_rng = np.random.default_rng(self.seed + 1_000_003)
        params = true_params or {
            name: float(test_rng.uniform(*self.parameter_ranges[name]))
            for name in self.parameter_names
        }
        corr = correlations or {
            name: float(test_rng.uniform(*bounds))
            for name, bounds in self.correlation_ranges.items()
        }
        return self._generate_record(
            params,
            corr,
            calibrate_aux=False,
            noise_std=noise_std,
        )