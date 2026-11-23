"""Synthetic epidemiological time-series and trajectory generation.

The four external features follow the construction described in the thesis:

    x_i = rho_i * y + sqrt(1 - rho_i**2) * u_i + epsilon_i,

where y and u_i are centered, unit-variance *sample* vectors,
<y, u_i> = 0, and epsilon_i contains independent Gaussian noise.

Note: because epsilon_i is added after the linear mixture, the empirical
Pearson correlation between x_i and y is not guaranteed to equal rho_i.
"""

from typing import Optional

import numpy as np
import torch
from torchdiffeq import odeint

from einn.ode.base_ode_model import BaseODEModel


class SyntheticDataGenerator:
    """Generate ODE trajectories, noisy observations, and correlated covariates."""

    def __init__(
        self,
        true_ode_model: BaseODEModel,
        seq_len: int = 200,
        rng: Optional[np.random.Generator] = None,
    ):
        self.true_ode_model = true_ode_model
        self.seq_len = int(seq_len)
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_trajectory(
        self,
        initial_states: list[float],
        true_params: list[float],
        seq_len: Optional[int] = None,
    ) -> np.ndarray:
        """Integrate the configured ODE model and return [time, state] states."""
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

        initial_tensor = torch.tensor(
            [initial_states], dtype=torch.float32
        )
        params_tensor = torch.tensor([true_params], dtype=torch.float32)
        t_eval = torch.arange(n_steps, dtype=torch.float32)

        def odefunc(t, states):
            return self.true_ode_model(states=states, params=params_tensor)

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
        """Center and standardize with population std (ddof=0).

        Non-constant vectors are required for a meaningful Pearson correlation.
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
        """Return a centered, unit-variance base vector orthogonal to y.

        Orthogonality is the sample inner product over the time axis.
        If the provided semantic base is (nearly) collinear with y, draw a
        replacement random base so that data generation can still proceed.
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

        # Try the intended semantic base first. Resampling is only needed
        # in the degenerate case where it is (almost) affine in y.
        for attempt in range(8):
            candidate = base if attempt == 0 else self.rng.normal(size=base.shape)
            v = np.asarray(candidate, dtype=np.float64).reshape(-1).copy()
            v -= v.mean()
            if np.std(v) <= 1e-12:
                continue

            # Gram--Schmidt: remove the orthogonal projection onto y.
            v -= (np.dot(v, y_flat) / norm_sq) * y_flat
            v -= v.mean()  # numerical protection against a small mean drift
            if np.std(v) <= 1e-10:
                continue

            # Centering and multiplication by a scalar preserve orthogonality.
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
        """Implement x = rho*y + sqrt(1-rho^2)*u + epsilon.

        y and u are standardized; <y,u>=0 to numerical precision.
        epsilon is sampled separately from N(0,extra_noise**2).
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
        """Generate one true trajectory, noisy observations, and four covariates."""
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

        target_idx = int(getattr(self.true_ode_model, "target_idx", states.shape[1] - 1))
        pure_y = states[:, target_idx : target_idx + 1]

        signal_peak = float(np.max(pure_y)) if float(np.max(pure_y)) > 0 else 1.0
        adjusted_noise_std = float(noise_std) * signal_peak
        noisy_y = pure_y + self.rng.normal(
            loc=0.0,
            scale=adjusted_noise_std,
            size=pure_y.shape,
        )
        noisy_y = np.clip(noisy_y, a_min=0.0, a_max=None)

        y_norm = self._standardize(pure_y)
        base_mobility = self.rng.normal(0.0, 1.0, size=(self.seq_len, 1))
        base_temp = np.sin(
            np.linspace(0.0, 4.0 * np.pi, self.seq_len)
        ).reshape(-1, 1)
        base_testing = np.linspace(
            0.0, 1.0, self.seq_len
        ).reshape(-1, 1)
        base_stringency = np.cumsum(
            self.rng.normal(0.0, 1.0, size=(self.seq_len, 1)),
            axis=0,
        )

        mobility = self._create_correlated_feature(
            base_mobility, y_norm, correlations["mobility"]
        )
        temperature = self._create_correlated_feature(
            base_temp, y_norm, correlations["temperature"]
        )
        testing = self._create_correlated_feature(
            base_testing, y_norm, correlations["testing"]
        )
        stringency = self._create_correlated_feature(
            base_stringency, y_norm, correlations["stringency"]
        )

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
