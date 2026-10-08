import numpy as np
import torch

from einn.ode.base_ode_model import BaseODEModel


class SyntheticDataGenerator:
    """
    Generates synthetic epidemiological time-series data using a specified ground-truth ODE model.
    It simulates real-world conditions by adding noise to the observable variable and 
    generating correlated exogenous features (e.g., mobility).
    """

    def __init__(self, true_ode_model: BaseODEModel, seq_len: int = 150):
        """
        Initializes the data generator.

        :param BaseODEModel true_ode_model: The TRUE underlying physics (e.g., SIR model). 
                                            This can be different from the model the EINN assumes!
        :param int seq_len: The total number of days/steps to simulate.
        """
        self.true_ode_model = true_ode_model
        self.seq_len = seq_len

    def generate(
            self,
            initial_states: list[float],
            true_params: list[float],
            noise_std: float = 0.05
    ) -> dict[str, np.ndarray]:
        """
        Runs a forward Euler simulation to generate the true epidemiological curves, 
        adds Gaussian noise to the observable target feature, and creates a correlated 
        exogenous feature set.

        :param list[float] initial_states: The starting values for the compartments (e.g., [0.99, 0.01, 0.0] for SIR).
        :param list[float] true_params: The ground truth physical parameters (e.g., [0.4, 0.1, 0.05]).
        :param float noise_std: Standard deviation of the Gaussian noise added to the target variable.
        :return dict: Dictionary containing the noisy inputs ('x_raw'), noisy targets ('y_raw'), 
                      and the hidden TRUE states ('true_latent_states').
        """
        d_s = self.true_ode_model.d_s

        if len(initial_states) != d_s:
            raise ValueError(f"Initial states length ({len(initial_states)}) must match ODE states ({d_s}).")

        states = np.zeros((self.seq_len, d_s))

        # Állapotok és paraméterek PyTorch tenzorként az ODE forward hívásához
        current_state = torch.tensor([initial_states], dtype=torch.float32)
        params_tensor = torch.tensor([true_params], dtype=torch.float32)

        # 1. Tiszta (zajmentes) szimuláció (Euler módszer)
        for t in range(self.seq_len):
            states[t] = current_state.detach().numpy()[0]
            # Kiszámoljuk a deriváltakat az igazi fizika alapján
            derivatives = self.true_ode_model(states=current_state, params=params_tensor)
            current_state = current_state + derivatives * 1.0

        # 2. A célváltozó kiválasztása a dinamikus target_idx alapján
        # Ha a BaseODEModel-ben kötelezővé tettük a target_idx-et, itt már biztosan létezik
        target_idx = self.true_ode_model.target_idx
        pure_y = states[:, target_idx:target_idx + 1]

        # 3. Zajosítás: A valós világban a mérések pontatlanok
        noisy_y = pure_y + np.random.normal(loc=0.0, scale=noise_std, size=pure_y.shape)
        # Megakadályozzuk a negatív esetszámokat (a biológiai realitás fenntartásához)
        noisy_y = np.clip(noisy_y, a_min=0.0, a_max=None)

        # 4. Külső adatok (exogenous features - X) generálása
        # Szimulálunk egy "mobilitási indexet", ami inverz korrelációban van az esetszámmal
        # (ha sok a beteg, csökken a mobilitás), és teszünk rá egy kis saját zajt is.
        dummy_mobility = np.ones((self.seq_len, 1)) - noisy_y * 0.5 + np.random.normal(0.0, 0.02, (self.seq_len, 1))

        # Az X bemenet tartalmazza a szimulált mobilitást ÉS a történelmi zajos I-t is
        noisy_x = np.concatenate([dummy_mobility, noisy_y], axis=1)

        return {
            "x_raw": noisy_x,
            "y_raw": noisy_y,
            "true_latent_states": states
        }
