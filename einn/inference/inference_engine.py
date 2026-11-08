import torch

from einn.data.data_scaling_stats import DataScalingStats
from einn.model.interface.einn_models import EINNModels


class InferenceEngine:
    """
    Handles future trajectory forecasting (extrapolation) using the trained EINN models.
    Operates strictly in evaluation mode without tracking gradients.

    Inference strictly relies on the Feature Module. It directly maps historical features and
    future time steps to future physical states.
    """

    def __init__(self, models: EINNModels):
        """
        Initializes the InferenceEngine with the trained model components.

        :param EINNModels models: The container holding all trained neural networks and the ODE model.
        """
        self.models = models

    def predict_future(
            self,
            historical_x: torch.Tensor,
            historical_x_mask: torch.Tensor,
            future_t: torch.Tensor,
            scaling_stats: DataScalingStats
    ) -> torch.Tensor:
        """
        Generates future predictions based on a window of historical data and future time steps.
        Uses the Feature Module to extract embeddings, and the Output Module to map them to states.

        :param torch.Tensor historical_x: The historical exogenous input features. Shape: [Batch, Seq_len, d_x].
        :param torch.Tensor historical_x_mask: The mask for the historical input window. Shape: [Batch, Seq_len].
        :param torch.Tensor future_t: The raw future time indices to predict for. Shape: [Batch, Future_steps, 1].
        :param DataScalingStats scaling_stats: Container holding the runtime scaling statistics (mean, std, min, max).
        :return torch.Tensor: The predicted future compartment states in the original physical scale.
                                      Shape: [Batch, Future_steps, d_s].
        """
        self._set_eval_mode()

        with torch.no_grad():
            # Feature Module obtains the embedding
            embedding = self.models.feature_module(
                x=historical_x,
                t=future_t,
                mask=historical_x_mask
            )

            # Output Module converts embeddings to state vectors
            states_prime = self.models.output_module(e=embedding)

            # Z-score inverse scaling
            if z_mean is not None and z_std is not None:
                z_mean_t = z_mean.to(states_prime.device)
                z_std_t = z_std.to(states_prime.device)
                states_prime = (states_prime * z_std_t) + z_mean_t

            return states_prime

    def _set_eval_mode(self):
        """
        Sets all model components to evaluation mode, disabling Dropout and BatchNorm updates.
        """
        self.models.time_module.eval()
        self.models.feature_module.eval()
        self.models.output_module.eval()
        self.models.ode_model.eval()
