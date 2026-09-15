import torch

from einn.config.einn_train_config import EINNTrainConfig
from einn.model.interface.phase_context import PhaseContext
from einn.model.interface.network_outputs import NetworkOutputs


class EINNForwardEngine:
    """
    Core execution engine for the EINN architecture.
    Responsible for routing data through the Time, Feature, and Output modules,
    computing empirical and analytical derivatives, and applying the gradient trick.
    """

    def __init__(self, train_config: EINNTrainConfig):
        """
        Initializes the Forward Engine with the training configuration.

        :param EINNTrainConfig train_config: Configuration containing training hyperparameters (e.g., device).
        """
        self.config = train_config

    @staticmethod
    def _get_jacobian_rows(outputs: torch.Tensor, inputs: torch.Tensor) -> list[torch.Tensor]:
        """
        Internal helper method to compute the gradients of each output feature w.r.t inputs.

        :param torch.Tensor outputs: The predicted tensor. Shape: [Batch, Seq_len, Features].
        :param torch.Tensor inputs: The input tensor (requires_grad=True).
        :return list[torch.Tensor]: A list of gradient tensors for each feature.
        """
        grads = []
        features_dim = outputs.shape[2]

        for i in range(features_dim):
            grad_i = torch.autograd.grad(
                outputs=outputs[:, :, i:i + 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(input=outputs[:, :, i:i + 1]),
                create_graph=True
            )[0]
            grads.append(grad_i)

        return grads
