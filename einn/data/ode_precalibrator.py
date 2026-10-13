import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from einn.ode.base_ode_model import BaseODEModel


class ODEPreCalibrator:
    """
    Generates a physically plausible 'aux_targets' (pre-calibrated latent curves) baseline.
    It fits static ODE parameters to the observed data while keeping initial states strictly locked.
    """

    def __init__(self, assumed_ode_model: BaseODEModel, d_p: int, device: str = 'cpu'):
        self.model = assumed_ode_model
        self.d_p = d_p
        self.device = torch.device(device)
        self.target_idx = self.model.target_idx

    @staticmethod
    def _inverse_sigmoid(val: float) -> float:
        """Helper to convert physical constraints (0-1) to logits."""
        val = max(min(val, 0.999), 0.001)  # Elkerüljük a végtelent
        return np.log(val / (1.0 - val))

    def calibrate(self, y_raw: np.ndarray, initial_states: list[float], epochs: int = 100,
                  lr: float = 0.1) -> np.ndarray:
        seq_len = len(y_raw)
        y_target = torch.tensor(y_raw, dtype=torch.float32, device=self.device)

        current_state_base = torch.tensor([initial_states], dtype=torch.float32, device=self.device)
        init_guess = [0.4, 0.2, 0.1, 0.05] if self.d_p == 4 else [0.4, 0.1]
        logits_guess = [self._inverse_sigmoid(v) for v in init_guess]

        params_logits = nn.Parameter(torch.tensor([logits_guess], device=self.device))

        # 1. ADAM HELYETT LBFGS OPTIMALIZÁLÓ!
        # Ez másodrendű deriváltakat használ, sokkal kevesebb epoch alatt (pl. 100) tökéletesebb eredményt ad.
        optimizer = torch.optim.LBFGS([params_logits], lr=lr, max_iter=20, line_search_fn="strong_wolfe")
        criterion = nn.MSELoss()

        print(f"Starting ODE Pre-calibration with LBFGS (d_s={self.model.d_s})...")

        best_states = None
        min_loss = float('inf')

        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                current_state = current_state_base.clone()
                current_params = torch.sigmoid(params_logits)
                simulated_states = []
                for t in range(seq_len):
                    simulated_states.append(current_state)
                    derivatives = self.model(states=current_state, params=current_params)
                    current_state = current_state + derivatives * 1.0

                simulated_states_tensor = torch.cat(simulated_states, dim=0)
                simulated_y = simulated_states_tensor[:, self.target_idx:self.target_idx + 1]
                loss = criterion(simulated_y, y_target)
                loss.backward()
                return loss

            # LBFGS lépés a closure függvénnyel
            loss = optimizer.step(closure)

            if loss.item() < min_loss:
                min_loss = loss.item()
                # Újraszámoljuk a legjobb állapotot mentéshez
                with torch.no_grad():
                    curr_state = current_state_base.clone()
                    curr_params = torch.sigmoid(params_logits)
                    sim_states = []
                    for t in range(seq_len):
                        sim_states.append(curr_state)
                        curr_state = curr_state + self.model(states=curr_state, params=curr_params) * 1.0
                    best_states = torch.cat(sim_states, dim=0).cpu().numpy()

        print(f"LBFGS Pre-calibration finished. Best MSE Loss: {min_loss:.4e}")
        return best_states
