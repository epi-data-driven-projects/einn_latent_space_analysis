import logging

import numpy as np
import torch
import torch.nn as nn

from torchdiffeq import odeint

from einn.ode.base_ode_model import BaseODEModel


class ODESolutionGenerator:
    """
    Generates a physically plausible 'aux_targets' (pre-calibrated latent curves) baseline.
    It fits static ODE parameters to the observed data while keeping initial states strictly locked,
    providing a biologically grounded starting point for the neural network.
    """

    def __init__(self, assumed_ode_model: BaseODEModel, d_p: int, device: str = 'cpu'):
        """
        Initializes the ODE solution generator.

        :param BaseODEModel assumed_ode_model: The mathematical ODE model (e.g., SIR, SEIRM) used to generate the curves
        :param int d_p: The number of physical parameters (e.g., 2 for SIR, 4 for SEIRM).
        :param str device: The computation device ('cpu' or 'cuda').
        """
        self.model = assumed_ode_model
        self.d_p = d_p
        self.device = torch.device(device)

        self.target_idx = self.model.target_idx
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _inverse_sigmoid(val: float) -> float:
        val = max(min(val, 0.999), 0.001)
        return float(np.log(val / (1.0 - val)))

    def generate(
        self,
        initial_states: list[float],
        true_params: list[float],
        seq_len: int
    ) -> np.ndarray:
        """
        Villámgyors ODE megoldás L-BFGS és for-ciklusok nélkül.
        Szintetikus adatoknál használatos, amikor a fizikai paraméterek ismertek.
        """
        # Idő-tenzor az odeint számára
        t_eval = torch.arange(seq_len, dtype=torch.float32, device=self.device)

        # Kezdeti állapot és paraméterek tenzorrá alakítása
        current_state = torch.tensor([initial_states], dtype=torch.float32, device=self.device)
        params_tensor = torch.tensor([true_params], dtype=torch.float32, device=self.device)

        # Wrapper függvény az odeint (t, y) aláírásához
        def odefunc(t, y):
            return self.model(states=y, params=params_tensor)

        with torch.no_grad():
            # Az odeint egyetlen lépésben megoldja a teljes differenciálegyenlet-rendszert!
            states_tensor = odeint(
                odefunc, 
                current_state, 
                t_eval, 
                method='rk4', 
                options={'step_size': 1.0}
            ).squeeze(1)

        return states_tensor.cpu().numpy()
