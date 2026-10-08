import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from einn.ode.base_ode_model import BaseODEModel


class ODEPreCalibrator:
    """
    Realistically calibrates an assumed ODE model to observed noisy data.
    Generates the 'aux_targets' (pre-calibrated latent curves) exactly as a
    traditional epidemiologist would before passing them to the EINN.
    """

    def __init__(self, assumed_ode_model: BaseODEModel, d_p: int, device: str = 'cpu'):
        """
        Initializes the pre-calibrator with the assumed physics model.

        :param BaseODEModel assumed_ode_model: The ODE model the EINN *believes* is true (Model B).
        :param int d_p: The number of physical parameters in the assumed model (e.g., 4 for SEIRM).
        :param str device: The computation device ('cpu' or 'cuda').
        """
        self.model = assumed_ode_model
        self.d_p = d_p
        self.device = torch.device(device)
        self.target_idx = self.model.target_idx

    def calibrate(self, y_raw: np.ndarray, epochs: int = 1000, lr: float = 0.01) -> np.ndarray:
        """
        Fits the assumed ODE model to the noisy observations using gradient descent.

        :param np.ndarray y_raw: The noisy observable data (e.g., infectious counts). Shape: [Seq_len, 1].
        :param int epochs: Number of optimization iterations.
        :param float lr: Learning rate for the Adam optimizer.
        :return np.ndarray: The simulated, best-fit full state curves to be used as aux_targets. Shape: [Seq_len, d_s].
        """
        seq_len = len(y_raw)
        y_target = torch.tensor(y_raw, dtype=torch.float32, device=self.device)

        # 1. Tanulható paraméterek definiálása (Kezdeti feltételek és ODE ráták)
        # Ezek "nyers" (logit) értékek, amiket majd transzformálunk a fizikai korlátok miatt
        init_states_logits = nn.Parameter(torch.zeros(1, self.model.d_s, device=self.device))
        params_logits = nn.Parameter(torch.zeros(1, self.d_p, device=self.device))

        optimizer = Adam([init_states_logits, params_logits], lr=lr)
        criterion = nn.MSELoss()

        best_states = None
        min_loss = float('inf')

        print(f"Starting ODE Pre-calibration for assumed model (d_s={self.model.d_s}, target_idx={self.target_idx})...")

        # 2. Optimalizációs ciklus (Görbeillesztés)
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Softmax: Biztosítja, hogy a kezdeti rekeszek összege pontosan 1.0 (100%) legyen
            current_state = torch.softmax(init_states_logits, dim=1)

            # Sigmoid: Biztosítja, hogy a fertőzési/gyógyulási ráták 0.0 és 1.0 között maradjanak
            current_params = torch.sigmoid(params_logits)

            simulated_states = []

            # Tiszta Euler integráció a teljes idősorra az aktuális paraméterekkel
            for t in range(seq_len):
                simulated_states.append(current_state)
                derivatives = self.model(states=current_state, params=current_params)
                current_state = current_state + derivatives * 1.0

            # Összefűzés: [Seq_len, d_s] méretű tenzor
            simulated_states = torch.cat(simulated_states, dim=0)

            # KIZÁRÓLAG a célváltozó (target_idx) alapján számolunk hibát a valós adathoz képest!
            simulated_y = simulated_states[:, self.target_idx:self.target_idx + 1]
            loss = criterion(simulated_y, y_target)

            loss.backward()
            optimizer.step()

            # A legjobb modellt elmentjük (legkisebb MSE)
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_states = simulated_states.detach().cpu().numpy()

            if (epoch + 1) % 200 == 0:
                print(f"Calibration Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4e}")

        print(f"Pre-calibration finished. Best MSE Loss: {min_loss:.4e}")

        return best_states
