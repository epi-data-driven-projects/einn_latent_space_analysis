"""
Synthetic epidemiological trajectory, noisy observation, and correlated covariate generator.
"""
from typing import Optional

import numpy as np
import torch
from torchdiffeq import odeint

from einn.ode.base_ode_model import BaseODEModel


class SyntheticDataGenerator:
    """
    Generate synthetic ODE trajectories, noisy target observations, and correlated covariates.

    Implements the controlled synthetic data pipeline described in thesis Section 2.3:
    numerically integrates a mechanistic epidemiological ODE with torchdiffeq.odeint,
    perturbs the target compartment with additive Gaussian noise, and constructs four
    exogenous covariates with prescribed Pearson correlations to the noise-free target.
    """

    def __init__(
        self,
        true_ode_model: BaseODEModel,
        seq_len: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialize the generator with a mechanistic ODE model and random number generator.

        :param BaseODEModel true_ode_model: Epidemiological ODE model defining ds/dt = f_ODE(s, Omega).
        :param int seq_len: Default number of discrete timesteps T per simulated trajectory.
        :param Optional[np.random.Generator] rng: NumPy RNG for reproducible noise and covariate sampling;
            creates a fresh default_rng() if None.
        """
        self.true_ode_model = true_ode_model
        self.seq_len = int(seq_len)
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_trajectory(
        self,
        initial_states: list[float],
        true_params: list[float],
        seq_len: Optional[int] = None,
    ) -> np.ndarray:
        """
        Numerically integrate the ODE model on t = 0, 1, ..., T - 1 using fixed-step RK4.

        :param list[float] initial_states: Initial compartment proportions s(0) of length d_s.
        :param list[float] true_params: Constant mechanistic ODE parameter vector Omega of length d_p.
        :param Optional[int] seq_len: Optional override for the number of timesteps T; defaults to self.seq_len.
        :return np.ndarray: Noise-free compartment trajectories of shape (T, d_s) (float32).
        :raises ValueError: If initial_states or true_params length mismatches the ODE model dimensions.
        """
        n_steps = self.seq_len if seq_len is None else int(seq_len)
        if len(initial_states) != self.true_ode_model.d_s:
            raise ValueError(
                f"Initial states length ({len(initial_states)}) must match "
                f"ODE states ({self.true_ode_model.d_s})."
            )

        expected_d_p = getattr(self.true_ode_model, "d_p", None)
        if expected_d_p is not None and len(true_params) != expected_d_p:
            raise ValueError(
                f"Expected {expected_d_p} parameters, got {len(true_params)}."
            )

        # Add a leading batch dimension (1, d_s) and (1, d_p) to match BaseODEModel's batched signature.
        initial_tensor = torch.tensor(
            [initial_states], dtype=torch.float32
        )
        params_tensor = torch.tensor([true_params], dtype=torch.float32)
        t_eval = torch.arange(n_steps, dtype=torch.float32)  # Shape: (T,)

        def odefunc(t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
            """
            Evaluate the ODE right-hand side f_ODE(states, params) for torchdiffeq.odeint.

            :param torch.Tensor t: Current scalar integration time (unused for autonomous ODEs).
            :param torch.Tensor states: Current compartment state tensor of shape (1, d_s).
            :return torch.Tensor: Time derivative ds/dt of shape (1, d_s).
            """
            return self.true_ode_model(states=states, params=params_tensor)

        # Integrate with RK4 (step_size=1.0); odeint returns (T, 1, d_s), squeezed to (T, d_s).
        with torch.no_grad():
            states = odeint(
                odefunc,
                initial_tensor,
                t_eval,
                method="rk4",
                options={"step_size": 1.0},
            ).squeeze(1)

        return states.cpu().numpy()

    @staticmethod
    def _standardize(values: np.ndarray) -> np.ndarray:
        """
        Standardize a time series to zero mean and unit population standard deviation (ddof=0).

        :param np.ndarray values: Input array of arbitrary shape (typically (T, 1)).
        :return np.ndarray: Standardized float64 array with the same shape as values.
        :raises ValueError: If the input is non-finite or nearly constant (std <= 1e-12).
        """
        x = np.asarray(values, dtype=np.float64)
        mean = np.mean(x)
        std = np.std(x)
        if not np.isfinite(std) or std <= 1e-12:
            raise ValueError(
                "Cannot standardize a (nearly) constant or non-finite time series."
            )
        return (x - mean) / std

    def _orthogonalized_semantic_base(
        self,
        semantic_base: np.ndarray,
        target_norm: np.ndarray,
    ) -> np.ndarray:
        """
        Construct a standardized covariate basis vector u orthogonal to the target y (<u, y> = 0).

        Applies Gram-Schmidt projection over the time axis. If semantic_base is nearly collinear
        with target_norm, retries up to 7 times with fresh standard normal draws.

        :param np.ndarray semantic_base: Raw covariate basis series of shape (T, 1).
        :param np.ndarray target_norm: Standardized noise-free target series y of shape (T, 1).
        :return np.ndarray: Standardized basis vector u of shape (T, 1) satisfying <u, y> = 0.
        :raises ValueError: If shapes mismatch, T < 3, target_norm has zero variance, or
            orthogonalization fails across all attempts.
        """
        y = np.asarray(target_norm, dtype=np.float64)
        base = np.asarray(semantic_base, dtype=np.float64)
        if base.shape != y.shape:
            raise ValueError(
                f"Semantic base and target must have the same shape: "
                f"{base.shape} vs {y.shape}."
            )
        if y.size < 3:
            raise ValueError(
                "At least three time points are needed for a nonzero vector "
                "orthogonal to both the constant vector and y."
            )

        y_flat = y.reshape(-1)
        norm_sq = float(np.dot(y_flat, y_flat))
        if norm_sq <= 1e-20:
            raise ValueError("Target time series has zero variance.")

        # Attempt Gram-Schmidt on the semantic base first; fall back to Gaussian noise if degenerate.
        for attempt in range(8):
            candidate = base if attempt == 0 else self.rng.normal(size=base.shape)
            v = np.asarray(candidate, dtype=np.float64).reshape(-1).copy()
            v -= v.mean()
            if np.std(v) <= 1e-12:
                continue

            # Subtract the projection of v onto y, then re-center to guard against floating-point drift.
            v -= (np.dot(v, y_flat) / norm_sq) * y_flat
            v -= v.mean()
            if np.std(v) <= 1e-10:
                continue

            # Standardization preserves both zero mean and orthogonality to y.
            u = self._standardize(v).reshape(base.shape)
            return u

        raise ValueError(
            "Could not construct an independent nonzero semantic component "
            "orthogonal to the target time series."
        )

    def _create_correlated_feature(
        self,
        semantic_base: np.ndarray,
        target_norm: np.ndarray,
        rho: float,
        extra_noise: float = 0.05,
    ) -> np.ndarray:
        """
        Mix a semantic basis with the target to achieve correlation rho plus additive noise.

        Implements x_i = rho * y + sqrt(1 - rho^2) * u + epsilon with epsilon ~ N(0, extra_noise^2),
        where y and u are standardized and orthogonal (thesis Section 2.3).

        :param np.ndarray semantic_base: Raw exogenous basis process of shape (T, 1).
        :param np.ndarray target_norm: Standardized noise-free target series y of shape (T, 1).
        :param float rho: Prescribed Pearson correlation coefficient in [-1, 1].
        :param float extra_noise: Standard deviation sigma_epsilon of additive Gaussian noise.
        :return np.ndarray: Correlated synthetic covariate series of shape (T, 1) (float64).
        :raises ValueError: If rho is outside [-1, 1] or extra_noise is negative/non-finite.
        """
        if not np.isfinite(rho) or not -1.0 <= rho <= 1.0:
            raise ValueError(f"Correlation must be in [-1, 1], got {rho}.")
        if not np.isfinite(extra_noise) or extra_noise < 0:
            raise ValueError("extra_noise must be finite and non-negative.")

        y = self._standardize(target_norm)
        u = self._orthogonalized_semantic_base(semantic_base, y)

        mixed = rho * y + np.sqrt(max(0.0, 1.0 - rho**2)) * u
        epsilon = self.rng.normal(0.0, extra_noise, size=mixed.shape)
        return mixed + epsilon

    def generate(
        self,
        initial_states: list[float],
        true_params: list[float],
        noise_std: float = 0.05,
        correlations: Optional[dict] = None,
    ) -> dict[str, np.ndarray]:
        """
        Simulate one ODE trajectory, perturb the target compartment, and generate 4 covariates.

        Assembles the 5-dimensional input matrix x_raw = [mobility, temperature, testing,
        stringency, noisy_y] of shape (T, 5) using the four base processes defined in thesis
        Section 2.3 (white noise, sinusoid, linear trend, and Gaussian random walk).

        :param list[float] initial_states: Initial compartment proportions s(0) of length d_s.
        :param list[float] true_params: Constant mechanistic ODE parameters Omega of length d_p.
        :param float noise_std: Relative Gaussian noise level scaled by the target peak value.
        :param Optional[dict] correlations: Mapping for keys 'mobility', 'temperature', 'testing',
            and 'stringency' to target correlations rho_i in [-1, 1]; uses defaults if None.
        :return dict[str, np.ndarray]: Dictionary with keys:
            'x_raw' of shape (T, 5),
            'y_raw' (noise-free target) of shape (T, 1),
            'y_noisy' (non-negative noisy target) of shape (T, 1), and
            'true_latent_states' (all ODE compartments) of shape (T, d_s).
        """
        # Numerically integrate the ground-truth ODE system: shape (T, d_s).
        states = self.generate_trajectory(
            initial_states=initial_states,
            true_params=true_params,
        )

        if correlations is None:
            correlations = {
                "mobility": -0.7,
                "temperature": -0.2,
                "testing": 0.8,
                "stringency": 0.9,
            }

        # Extract the clean target compartment of shape (T, 1) and add peak-scaled Gaussian noise.
        target_idx = int(getattr(self.true_ode_model, "target_idx", states.shape[1] - 1))
        pure_y = states[:, target_idx : target_idx + 1]

        signal_peak = float(np.max(pure_y)) if float(np.max(pure_y)) > 0 else 1.0
        adjusted_noise_std = float(noise_std) * signal_peak
        noisy_y = pure_y + self.rng.normal(
            loc=0.0,
            scale=adjusted_noise_std,
            size=pure_y.shape,
        )
        # Clip noisy observations at zero to preserve epidemiological non-negativity.
        noisy_y = np.clip(noisy_y, a_min=0.0, a_max=None)

        # Construct the four exogenous base processes of shape (T, 1) (thesis Section 2.3):
        # white noise (mobility), periodic sinusoid (temperature), linear trend (testing),
        # and cumulative Gaussian random walk (stringency).
        y_norm = self._standardize(pure_y)
        base_mobility = self.rng.normal(0.0, 1.0, size=(self.seq_len, 1))
        base_temp = np.sin(
            np.linspace(0.0, 4.0 * np.pi, self.seq_len)
        ).reshape(-1, 1)
        base_testing = np.linspace(
            0.0, 1.0, self.seq_len
        ).reshape(-1, 1)
        base_stringency = np.cumsum(
            self.rng.normal(loc=0.0, scale=1.0, size=(self.seq_len, 1)),
            axis=0,
        )

        # Mix each orthogonalized base process with y_norm to enforce the requested correlations.
        mobility = self._create_correlated_feature(
            semantic_base=base_mobility, target_norm=y_norm, rho=correlations["mobility"], extra_noise=noise_std
        )
        temperature = self._create_correlated_feature(
            semantic_base=base_temp, target_norm=y_norm, rho=correlations["temperature"], extra_noise=noise_std
        )
        testing = self._create_correlated_feature(
            semantic_base=base_testing, target_norm=y_norm, rho=correlations["testing"], extra_noise=noise_std
        )
        stringency = self._create_correlated_feature(
            semantic_base=base_stringency, target_norm=y_norm, rho=correlations["stringency"], extra_noise=noise_std
        )

        # Concatenate the 4 correlated covariates and the noisy target observation: shape (T, 5).
        x_raw = np.concatenate(
            [mobility, temperature, testing, stringency, noisy_y],
            axis=1,
        )

        return {
            "x_raw": x_raw,
            "y_raw": pure_y,
            "y_noisy": noisy_y,
            "true_latent_states": states,
        }
