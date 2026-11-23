
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
        """Initializes the ODE solution generator.

        :param BaseODEModel assumed_ode_model: The mathematical ODE model (e.g., SIR, SEIRM) used to generate the curves
        :param int d_p: The number of physical parameters (e.g., 2 for SIR, 4 for SEIRM).
        :param str device: The computation device ('cpu' or 'cuda').

        :return None: No return value; state or serialized artifacts are updated in place.
        """
        self.model = assumed_ode_model
        self.d_p = d_p
        self.device = torch.device(device)

        self.target_idx = self.model.target_idx
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _inverse_sigmoid(val: float) -> float:
        """
        Transform a bounded physical parameter into its unconstrained logit.

        :param float val: Physical initial parameter value converted into an unconstrained representation.
        :return float: Computed result in the configured output representation.
        """
        val = max(min(val, 0.999), 0.001)
        return float(np.log(val / (1.0 - val)))

    def generate(
        self,
        initial_states: list[float],
        true_params: list[float],
        seq_len: int
    ) -> np.ndarray:
        """Integrate the configured ODE system and return its reference trajectories without parameter optimization.


        :param list[float] initial_states: Initial compartment populations in model state order.
        :param list[float] true_params: Physical ODE parameter vector used to simulate the ground-truth trajectory.
        :param int seq_len: Number of time coordinates per generated trajectory.
        :return np.ndarray: Computed result in the configured output representation.
        """
        # Construct the time grid for numerical integration.
        t_eval = torch.arange(seq_len, dtype=torch.float32, device=self.device)

        # Convert initial conditions and parameters to tensors.
        current_state = torch.tensor([initial_states], dtype=torch.float32, device=self.device)
        params_tensor = torch.tensor([true_params], dtype=torch.float32, device=self.device)

        # Adapt to the odeint(t, y) callback signature.
        def odefunc(t, y):
            """Return mechanistic compartment derivatives for a numerical ODE solver step.


            :param Any t: Time coordinates of the modeled trajectory.
            :param Any y: Observed epidemiological target values.
            :return Any: The computed objects, metrics, or artifact paths for the requested operation.
            """
            return self.model(states=y, params=params_tensor)

        with torch.no_grad():
            # Integrate the full differential-equation system.
            states_tensor = odeint(
                odefunc, 
                current_state, 
                t_eval, 
                method='rk4', 
                options={'step_size': 1.0}
            ).squeeze(1)

        return states_tensor.cpu().numpy()
