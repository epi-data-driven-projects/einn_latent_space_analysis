import numpy as np
import torch
from einn.ode.base_ode_model import BaseODEModel

class SyntheticDataGenerator:
    def __init__(self, true_ode_model: BaseODEModel, seq_len: int = 300): # Megnövelt seq_len!
        self.true_ode_model = true_ode_model
        self.seq_len = seq_len

    def generate(self, initial_states: list[float], true_params: list[float], noise_std: float = 0.05) -> dict[str, np.ndarray]:
        d_s = self.true_ode_model.d_s
        if len(initial_states) != d_s:
            raise ValueError(f"Initial states length ({len(initial_states)}) must match ODE states ({d_s}).")

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

        # 3. TÖBB EXOGÉN ADAT GENERÁLÁSA (d_x = 5 lesz a zajos Y-nal együtt)
        # a) Mobilitás (inverz a fertőzöttekkel)
        mobility = np.ones((self.seq_len, 1)) - noisy_y * 0.5 + np.random.normal(0, 0.02, (self.seq_len, 1))
        # b) Hőmérséklet (szinuszos szezonalitás)
        temperature = np.sin(np.linspace(0, 4 * np.pi, self.seq_len)).reshape(-1, 1) * 0.2 + 0.5 + np.random.normal(0, 0.02, (self.seq_len, 1))
        # c) Tesztelési kapacitás (lineárisan növekvő)
        testing_cap = np.linspace(0.1, 0.9, self.seq_len).reshape(-1, 1) + np.random.normal(0, 0.01, (self.seq_len, 1))
        # d) Szigorúsági index (véletlenszerű lépések - random walk)
        stringency = np.cumsum(np.random.normal(0, 0.01, (self.seq_len, 1)), axis=0)
        stringency = (stringency - stringency.min()) / (stringency.max() - stringency.min() + 1e-8)

        # Bemeneti mátrix (X) összefűzése
        noisy_x = np.concatenate([mobility, temperature, testing_cap, stringency, noisy_y], axis=1)

        return {
            "x_raw": noisy_x,
            "y_raw": noisy_y,
            "true_latent_states": states
        }
