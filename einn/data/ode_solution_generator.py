import logging

import numpy as np
import torch
import torch.nn as nn

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
        """
        Helper method to map strictly constrained physical probabilities (0 to 1)
        back into an unbounded real number space (-inf to +inf).
        This allows the optimizer to explore the parameter space freely without violating biological bounds.

        :param float val: The target physical value (must be strictly between 0 and 1).
        :return float: The unconstrained logit value.
        """
        # Clamping prevents math domain errors (log of zero or division by zero)
        val = max(min(val, 0.999), 0.001)
        return float(np.log(val / (1.0 - val)))

    def calibrate(
            self,
            y_raw: np.ndarray,
            initial_states: list[float],
            initial_guess: list[float] | None = None,
            epochs: int = 15,
            lr: float = 0.1
    ) -> np.ndarray:
        """
        Executes the optimization loop to find the best-fitting static physical parameters for the observed data,
        and returns the full pre-calibrated latent state trajectory.

        :param np.ndarray y_raw: The observed target time-series data. Shape: [Seq_len, 1].
        :param list[float] initial_states: The strictly locked initial compartment fractions at t=0.
        :param list[float] | None initial_guess: Optional starting values for the parameters.
        :param int epochs: Number of macroscopic optimization steps.
        :param float lr: Learning rate for the L-BFGS optimizer.
        :return np.ndarray: The pre-calibrated latent trajectories for all compartments. Shape: [Seq_len, d_s].
        """
        seq_len = len(y_raw)
        y_target = torch.tensor(y_raw, dtype=torch.float32, device=self.device)

        # The initial state is locked and reshaped to simulate a batch size of 1 [1, d_s]
        current_state_base = torch.tensor([initial_states], dtype=torch.float32, device=self.device)

        # Fallback to standard initialization if no specific guess is provided
        if initial_guess is None:
            initial_guess = [0.4, 0.2, 0.1, 0.05][:self.d_p]

        if len(initial_guess) != self.d_p:
            raise ValueError(f"Length of initial_guess ({len(initial_guess)}) must match d_p ({self.d_p}).")

        # Convert physical guesses into unbounded logits for optimization
        logits_guess = [self._inverse_sigmoid(v) for v in initial_guess]
        params_logits = nn.Parameter(torch.tensor([logits_guess], device=self.device))

        optimizer = torch.optim.LBFGS([params_logits], lr=lr, max_iter=20, line_search_fn="strong_wolfe")
        criterion = nn.MSELoss()

        self.logger.info(f"Starting ODE Pre-calibration using L-BFGS (d_s={self.model.d_s}, d_p={self.d_p})...")

        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()

                current_state = current_state_base.clone()
                # Apply sigmoid to constrain the unbounded logits back to valid physical rates [0, 1]
                current_params = torch.sigmoid(params_logits)

                simulated_states_ = []

                # Forward Euler integration loop
                for _ in range(seq_len):
                    simulated_states_.append(current_state)
                    derivatives = self.model(states=current_state, params=current_params)
                    current_state = current_state + derivatives * 1.0

                # Concatenate list of [1, d_s] tensors into [Seq_len, d_s]
                simulated_states_tensor = torch.cat(simulated_states_, dim=0)

                # Extract only the targeted compartment for loss calculation
                simulated_y = simulated_states_tensor[:, self.target_idx:self.target_idx + 1]

                loss = criterion(simulated_y, y_target)
                loss.backward()

                return loss

            # L-BFGS manages multiple internal iterations per single step call
            optimizer.step(closure)

        # Post-optimization trajectory generation
        # L-BFGS inherently stops at the optimal parameter set. We extract these optimized parameters
        # and run one final, detached forward pass to generate the baseline latent curves.
        with torch.no_grad():
            final_state = current_state_base.clone()
            final_params = torch.sigmoid(params_logits)

            self.logger.info(
                f"Calibration successful. Final physical parameters: {final_params.cpu().numpy().round(4)}")

            simulated_states = []
            for _ in range(seq_len):
                simulated_states.append(final_state)
                final_state = final_state + self.model(states=final_state, params=final_params) * 1.0

            best_states = torch.cat(simulated_states, dim=0).cpu().numpy()

        return best_states
