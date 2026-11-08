import numpy as np
import torch

from einn.ode.base_ode_model import BaseODEModel


class SyntheticDataGenerator:
    """
    Generates synthetic epidemiological time-series data for training and evaluating the EINN models.
    It simulates pure physical trajectories using an ODE model and generates noisy, correlated
    exogenous features (e.g., mobility, temperature) to mimic real-world latent space complexity.
    """

    def __init__(self, true_ode_model: BaseODEModel, seq_len: int = 200):
        """
        Initializes the synthetic data generator.

        :param BaseODEModel true_ode_model: The mathematical ODE model used to generate the ground-truth dynamics.
        :param int seq_len: The total number of time steps to simulate in the generated trajectory.
        """
        self.true_ode_model = true_ode_model
        self.seq_len = seq_len

    @staticmethod
    def _create_correlated_feature(
            semantic_base: np.ndarray,
            target_norm: np.ndarray,
            rho: float,
            extra_noise: float = 0.05
    ) -> np.ndarray:
        """
        Generates a synthetic feature that is mathematically correlated with a target sequence.

        :param np.ndarray semantic_base: The base underlying signal (e.g., sine wave, random walk). Shape: [Seq_len, 1].
        :param np.ndarray target_norm: The normalized target variable to correlate with. Shape: [Seq_len, 1].
        :param float rho: The desired Pearson correlation coefficient (between -1.0 and 1.0).
        :param float extra_noise: The standard deviation of the independent Gaussian noise added for realism.
        :return np.ndarray: The mixed, correlated exogenous feature. Shape: [Seq_len, 1].
        """
        # Standardize the base signal to ensure mathematical stability during mixing
        s_norm = (semantic_base - np.mean(semantic_base)) / (np.std(semantic_base) + 1e-8)

        # Mix the target and the base signal based on the target correlation coefficient (rho)
        mixed = rho * target_norm + np.sqrt(1 - rho ** 2) * s_norm

        # Inject an independent layer of Gaussian noise to simulate real-world measurement imperfections
        mixed += np.random.normal(loc=0.0, scale=extra_noise, size=mixed.shape)

        return mixed

    def generate(
            self,
            initial_states: list[float],
            true_params: list[float],
            noise_std: float = 0.05,
            correlations: dict = None
    ) -> dict[str, np.ndarray]:
        """
        Executes the full generation pipeline: simulates ODE dynamics, applies noise,
        and constructs correlated exogenous inputs.

        :param list[float] initial_states: The starting compartment fractions (must sum to <= 1.0). Length: d_s.
        :param list[float] true_params: The static physical parameters driving the ODE (e.g., beta, gamma).
        :param float noise_std: Standard deviation of the Gaussian noise applied to the observed target.
        :param dict correlations: Dictionary defining the desired correlation coefficients for exogenous features.
        :return dict[str, np.ndarray]: A dictionary containing raw inputs (X), raw targets (y), and true latent states.
        """
        d_s = self.true_ode_model.d_s
        if len(initial_states) != d_s:
            raise ValueError(f"Initial states length ({len(initial_states)}) must match ODE states ({d_s}).")

        # Establish default biological and sociological correlations if none are provided.
        # - Mobility: Strong negative (people stay home during infection peaks)
        # - Temperature: Mild negative (seasonal dampening effect)
        # - Testing: Strong positive (testing capacity ramps up during outbreaks)
        # - Stringency: Very strong positive (governments lock down following cases)
        if correlations is None:
            correlations = {
                'mobility': -0.7,
                'temperature': -0.2,
                'testing': 0.8,
                'stringency': 0.9
            }

        states = np.zeros((self.seq_len, d_s))
        current_state = torch.tensor([initial_states], dtype=torch.float32)
        params_tensor = torch.tensor([true_params], dtype=torch.float32)

        # Simulate the pure compartmental dynamics forward in time using the Euler integration method.
        for t in range(self.seq_len):
            states[t] = current_state.detach().numpy()[0]
            derivatives = self.true_ode_model(states=current_state, params=params_tensor)
            current_state = current_state + derivatives * 1.0

        # Extract the pure target compartment (e.g., Infected) and overlay observational Gaussian noise.
        # Clipping at 0.0 ensures biological validity (you cannot have negative infected individuals).
        target_idx = getattr(self.true_ode_model, 'target_idx', d_s - 1)
        pure_y = states[:, target_idx:target_idx + 1]
        noisy_y = pure_y + np.random.normal(loc=0.0, scale=noise_std, size=pure_y.shape)
        noisy_y = np.clip(noisy_y, a_min=0.0, a_max=None)

        # Standardize the pure target trajectory. This serves as the 'anchor' signal
        # to pull the exogenous features into correlation.
        y_norm = (pure_y - np.mean(pure_y)) / (np.std(pure_y) + 1e-8)

        # Generate structural base signals for the exogenous features before applying correlations
        base_mobility = np.random.normal(loc=0.0, scale=1.0, size=(self.seq_len, 1))

        base_temp = np.sin(np.linspace(0, 4 * np.pi, self.seq_len)).reshape(-1, 1)

        base_testing = np.linspace(0, 1, self.seq_len).reshape(-1, 1)

        base_stringency = np.cumsum(np.random.normal(loc=0.0, scale=1.0, size=(self.seq_len, 1)), axis=0)

        # Mix the base signals with the normalized target trajectory based on requested rho values
        mobility = self._create_correlated_feature(base_mobility, target_norm=y_norm,
                                                   rho=correlations['mobility'])
        temperature = self._create_correlated_feature(base_temp, target_norm=y_norm,
                                                      rho=correlations['temperature'])
        testing = self._create_correlated_feature(base_testing, target_norm=y_norm,
                                                  rho=correlations['testing'])
        stringency = self._create_correlated_feature(base_stringency, target_norm=y_norm,
                                                     rho=correlations['stringency'])

        # Assemble the final observation matrix X.
        noisy_x = np.concatenate([mobility, temperature, testing, stringency, noisy_y], axis=1)

        return {
            "x_raw": noisy_x,
            "y_raw": noisy_y,
            "true_latent_states": states
        }
