import numpy as np
import torch
from einn.ode.base_ode_model import BaseODEModel

class SyntheticDataGenerator:
    def __init__(self, true_ode_model: BaseODEModel, seq_len: int = 200):
        self.true_ode_model = true_ode_model
        self.seq_len = seq_len

    def generate(self, initial_states: list[float], true_params: list[float], noise_std: float = 0.05, correlations: dict = None) -> dict[str, np.ndarray]:
        d_s = self.true_ode_model.d_s
        if len(initial_states) != d_s:
            raise ValueError(f"Initial states length ({len(initial_states)}) must match ODE states ({d_s}).")

        # Alapértelmezett korrelációk, ha nem adunk meg mást
        if correlations is None:
            correlations = {
                'mobility': -0.7,      # Erős negatív korreláció a fertőzöttekkel (sokan betegek = otthon maradnak)
                'temperature': -0.2,   # Enyhe negatív (szezonális hatás)
                'testing': 0.8,        # Erős pozitív (felfutó ágban több a teszt)
                'stringency': 0.9      # Nagyon erős pozitív (lezárások követik a csúcsot)
            }

        states = np.zeros((self.seq_len, d_s))
        current_state = torch.tensor([initial_states], dtype=torch.float32)
        params_tensor = torch.tensor([true_params], dtype=torch.float32)

        # 1. Tiszta szimuláció
        for t in range(self.seq_len):
            states[t] = current_state.detach().numpy()[0]
            derivatives = self.true_ode_model(states=current_state, params=params_tensor)
            current_state = current_state + derivatives * 1.0

        # 2. Célváltozó és zaj
        target_idx = getattr(self.true_ode_model, 'target_idx', d_s - 1)
        pure_y = states[:, target_idx:target_idx+1]
        noisy_y = pure_y + np.random.normal(loc=0.0, scale=noise_std, size=pure_y.shape)
        noisy_y = np.clip(noisy_y, a_min=0.0, a_max=None)

        # 3. KORRELÁLT EXOGÉN ADATOK GENERÁLÁSA
        # Normalizáljuk a tiszta Y-t a matematikai keveréshez
        y_norm = (pure_y - np.mean(pure_y)) / (np.std(pure_y) + 1e-8)

        def create_correlated_feature(semantic_base: np.ndarray, rho: float, extra_noise: float = 0.05) -> np.ndarray:
            """
            Létrehoz egy adott 'rho' korrelációjú jelet a fertőzötti görbével (y_norm).
            """
            s_norm = (semantic_base - np.mean(semantic_base)) / (np.std(semantic_base) + 1e-8)
            # A statisztikai keverés képlete:
            mixed = rho * y_norm + np.sqrt(1 - rho**2) * s_norm
            # Plusz egy kis valósághű független zaj
            mixed += np.random.normal(0, extra_noise, size=mixed.shape)
            return mixed

        # a) Mobilitás (alapja sima véletlen zaj, amit a fertőzés görbéje inverz módon húz el)
        base_mobility = np.random.normal(0, 1, (self.seq_len, 1))
        mobility = create_correlated_feature(base_mobility, rho=correlations['mobility'])

        # b) Hőmérséklet (szinuszos szezonalitás az alapja)
        base_temp = np.sin(np.linspace(0, 4 * np.pi, self.seq_len)).reshape(-1, 1)
        temperature = create_correlated_feature(base_temp, rho=correlations['temperature'])

        # c) Tesztelési kapacitás (lineárisan növekvő trend)
        base_testing = np.linspace(0, 1, self.seq_len).reshape(-1, 1)
        testing = create_correlated_feature(base_testing, rho=correlations['testing'])

        # d) Szigorúsági index (random walk / halmozott lépések)
        base_stringency = np.cumsum(np.random.normal(0, 1, (self.seq_len, 1)), axis=0)
        stringency = create_correlated_feature(base_stringency, rho=correlations['stringency'])

        # Bemeneti mátrix (X) összefűzése: [Mobilitás, Hőmérséklet, Tesztelés, Szigorúság, Zajos Y]
        noisy_x = np.concatenate([mobility, temperature, testing, stringency, noisy_y], axis=1)

        return {
            "x_raw": noisy_x,
            "y_raw": noisy_y,
            "true_latent_states": states
        }
