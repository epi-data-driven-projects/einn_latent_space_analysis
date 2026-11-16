from typing import Optional

import numpy as np
import torch
from torchdiffeq import odeint

from einn.ode.base_ode_model import BaseODEModel


class SyntheticDataGenerator:
    """
    Generate ODE trajectories, noisy observations, and correlated covariates.

    This class creates synthetic datasets for epidemiological modeling. It uses
    a specified ODE model to generate ground-truth epidemic curves and then
    constructs a realistic observed dataset by adding Gaussian noise and generating
    synthetic socio-economic or environmental covariates (e.g., mobility, temperature)
    that correlate with the primary epidemic signal.
    """

    def __init__(
        self,
        true_ode_model: BaseODEModel,
        seq_len: int = 200,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initializes the Synthetic Data Generator.

        :param BaseODEModel true_ode_model: The mathematical ODE model used to
            generate the true latent states (e.g., SIR, SIS).
        :param int seq_len: The sequence length (number of time steps) for the
            generated time series. Defaults to 200.
        :param Optional[np.random.Generator] rng: A NumPy random number generator
            instance for reproducible noise and covariate generation. If None, a
            default generator is instantiated.
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
        Integrate the configured ODE model and return the continuous states over time.

        This method solves the initial value problem for the provided ODE model
        using the 4th-order Runge-Kutta (RK4) method.

        :param list[float] initial_states: The starting values for each compartment
            in the ODE model at time t=0. Length must match `true_ode_model.d_s`.
        :param list[float] true_params: The physical parameters governing the ODE
            dynamics (e.g., beta, gamma). Length must match expected parameters.
        :param Optional[int] seq_len: The number of time steps to integrate. If
            None, defaults to `self.seq_len`.

        :return np.ndarray: A 2D array of shape (seq_len, d_s) containing the
            computed trajectory for all ODE compartments.
        :raises ValueError: If the length of `initial_states` or `true_params`
            does not match the model's expected dimensions.
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
        """
        Applies Z-score standardization to a numpy array.

        :param np.ndarray values: The array to standardize.
        :return np.ndarray: The standardized array with mean 0 and standard deviation 1.
        """
        return (values - np.mean(values)) / (np.std(values) + 1e-8)

    def _create_correlated_feature(
        self,
        semantic_base: np.ndarray,
        target_norm: np.ndarray,
        rho: float,
        extra_noise: float = 0.05,
    ) -> np.ndarray:
        """
        Creates a synthetic covariate that correlates with the target epidemic signal.

        Blends a baseline semantic signal (e.g., random walk, sine wave) with the
        standardized epidemic curve to achieve a specific Pearson correlation coefficient.

        :param np.ndarray semantic_base: The base pattern for the covariate (e.g., sine wave).
        :param np.ndarray target_norm: The standardized target epidemic signal (e.g., infected curve).
        :param float rho: The desired correlation coefficient (-1.0 to 1.0) between the
            generated feature and the target signal.
        :param float extra_noise: The standard deviation of the additive Gaussian noise.
            Defaults to 0.05.

        :return np.ndarray: The generated synthetic covariate array.
        :raises ValueError: If `rho` is not within the [-1, 1] range.
        """
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"Correlation must be in [-1, 1], got {rho}.")

        semantic_norm = self._standardize(semantic_base)
        mixed = rho * target_norm + np.sqrt(max(0.0, 1.0 - rho**2)) * semantic_norm
        mixed += self.rng.normal(0.0, extra_noise, size=mixed.shape)
        return mixed

    def generate(
        self,
        initial_states: list[float],
        true_params: list[float],
        noise_std: float = 0.05,
        correlations: Optional[dict] = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate one true trajectory, noisy observations, and four covariates.

        This method acts as the primary data synthesis pipeline. It integrates the ODE,
        extracts the target compartment, adds proportional Gaussian noise to simulate
        real-world measurement errors, and creates four correlated features
        (mobility, temperature, testing, stringency).

        :param list[float] initial_states: The starting compartment values at t=0.
        :param list[float] true_params: The physical ODE parameters.
        :param float noise_std: The baseline standard deviation for the observational noise,
            which is scaled relative to the peak of the target signal. Defaults to 0.05.
        :param Optional[dict] correlations: A dictionary specifying the desired correlation
            for each covariate. If None, default socio-economic correlations are used.

        :return dict[str, np.ndarray]: A dictionary containing:
            - 'x_raw': The concatenated (seq_len, 5) matrix of covariates and noisy observations.
            - 'y_raw': The pristine (seq_len, 1) target trajectory.
            - 'y_noisy': The noisy (seq_len, 1) target observations (clipped at 0).
            - 'true_latent_states': The full (seq_len, d_s) pristine ODE trajectory.
        """
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
