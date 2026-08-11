import torch

from einn.model.interface.einn_models import EINNModels


class InferenceEngine:
    """
    Engine responsible for making future predictions using the trained EINN models.
    """

    def __init__(self, models: EINNModels, device: str = 'cpu'):
        """
        Initializes the InferenceEngine.

        Args:
            models (EINNModels): Dataclass containing the trained Feature, Time, Output, and ODE models.
            device (str): The computing device ('cpu' or 'cuda').
        """
        self.models = models
        self.device = torch.device(device)

    def predict(self, x: torch.Tensor, future_t: torch.Tensor) -> torch.Tensor:
        """
        Predicts future ODE compartment states based on past observations.

        Args:
            x (torch.Tensor): Past observed context. Shape: [Batch, Past_seq_len, d_x].
            future_t (torch.Tensor): Future time steps to predict. Shape: [Batch, Future_seq_len, 1].

        Returns:
            torch.Tensor: Predicted compartment states. Shape: [Batch, Future_seq_len, d_s].
        """
        # Ensure inputs are on the correct device (GPU)
        x = x.to(self.device)
        future_t = future_t.to(self.device)

        # Set neural networks to evaluation mode (disables Dropout and fixes BatchNorm layers)
        self.models.feature_module.set_eval_mode()
        self.models.output_module.set_eval_mode()

        # Disable gradient calculation to save memory and drastically speed up inference
        with torch.no_grad():
            # 1. Encode past context (x) and decode into future time steps (future_t)
            # e_future_F shape: [Batch, Future_seq_len, d_e]
            e_future_F = self.models.feature_module(x, future_t)

            # 2. Map the latent embeddings to physical ODE states
            # Note: OutputModule is currently configured to output ONLY the states (d_s)
            # s_future_F shape: [Batch, Future_seq_len, d_s]
            s_future_F = self.models.output_module(e_future_F)

        return s_future_F
