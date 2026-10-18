import numpy as np
import torch
from einn.ode.base_ode_model import BaseODEModel


class SyntheticDataGenerator:
    def __init__(self, true_ode_model: BaseODEModel, seq_len: int = 300):
        self.true_ode_model = true_ode_model
        self.seq_len = seq_len

    def _generate_correlated_series(self, target: np.ndarray, rho: float) -> np.ndarray:
        """
        Generál egy [0, 1] közé skálázott adatsort, ami pontosan 'rho'
        Pearson-korrelációban van a 'target' vektorral.
        """
        target_flat = target.flatten()
        # Z-score a bázisra
        t_std = (target_flat - np.mean(target_flat)) / (np.std(target_flat) + 1e-8)

        # Független zaj generálása
        noise = np.random.normal(0, 1, size=len(t_std))

        # Gram-Schmidt: Eltávolítjuk a target irányát a zajból, hogy teljesen ortogonális legyen
        noise = noise - np.dot(t_std, noise) / np.dot(t_std, t_std) * t_std
        noise = noise / (np.std(noise) + 1e-8)

        # Vektorok lineáris kombinációja a kívánt rho korrelációval
        correlated_flat = rho * t_std + np.sqrt(1 - rho ** 2) * noise

        # Visszaskálázás [0, 1] tartományba a reálisabb exogén értékekhez
        correlated_flat = (correlated_flat - np.min(correlated_flat)) / (np.ptp(correlated_flat) + 1e-8)
        return correlated_flat.reshape(-1, 1)

    def generate(self, initial_states: list[float], true_params: list[float], noise_std: float = 0.05,
                 observable_state_indices: list[int] = None) -> dict[str, np.ndarray]:
        d_s = self.true_ode_model.d_s
        states = np.zeros((self.seq_len, d_s))
        current_state = torch.tensor([initial_states], dtype=torch.float32)
        params_tensor = torch.tensor([true_params], dtype=torch.float32)

        # 1. Tiszta szimuláció
        for t in range(self.seq_len):
            states[t] = current_state.detach().numpy()[0]
            current_state = current_state + self.true_ode_model(states=current_state, params=params_tensor) * 1.0

        # 2. Célváltozó (Y) - Fertőzöttek (I) indexe általában 2 a SEIRM-ben, 1 a SIR-ben
        target_idx = getattr(self.true_ode_model, 'target_idx', d_s - 1)
        pure_y = states[:, target_idx:target_idx + 1]
        noisy_y = pure_y + np.random.normal(0.0, noise_std, pure_y.shape)
        noisy_y = np.clip(noisy_y, a_min=0.0, a_max=None)

        # Bázis a korrelációkhoz (legyen az I rekesz, ha van, különben a célváltozó)
        # Bázis a korrelációkhoz (Ha d_s == 3, akkor SIR modell és az 'I' az 1. index, különben SEIRM és a 2. index)
        base_for_corr = states[:, 1:2] if self.true_ode_model.d_s == 3 else states[:, 2:3]
        # 3. Exogén (külső) jellemzők generálása VALÓS KORRELÁCIÓVAL
        # - Mobilitás: Erősen negatív korreláció a fertőzöttekkel (ha sok a beteg, otthon maradnak)
        mobility = self._generate_correlated_series(base_for_corr, rho=-0.85)

        # - Szigorúság (Stringency): Erősen pozitív (sok beteg -> szigorúbb lezárások)
        stringency = self._generate_correlated_series(base_for_corr, rho=0.80)

        # - Tesztelési kapacitás: Gyengén pozitív (lassan reagál a hullámra)
        testing_cap = self._generate_correlated_series(base_for_corr, rho=0.40)

        # - Hőmérséklet: Szezonális, maradhat független szinusz
        temperature = np.sin(np.linspace(0, 4 * np.pi, self.seq_len)).reshape(-1, 1) * 0.2 + 0.5 + np.random.normal(0,
                                                                                                                    0.02,
                                                                                                                    (self.seq_len,
                                                                                                                     1))

        base_features = [mobility, temperature, testing_cap, stringency]

        # 4. Dinamikus megfigyelt állapotok
        if observable_state_indices is not None:
            for idx in observable_state_indices:
                obs_pure = states[:, idx:idx + 1]
                obs_noisy = obs_pure + np.random.normal(0.0, noise_std, obs_pure.shape)
                base_features.append(np.clip(obs_noisy, a_min=0.0, a_max=None))
        else:
            base_features.append(noisy_y)

        noisy_x = np.concatenate(base_features, axis=1)

        return {
            "x_raw": noisy_x,
            "y_raw": noisy_y,
            "true_latent_states": states
        }
