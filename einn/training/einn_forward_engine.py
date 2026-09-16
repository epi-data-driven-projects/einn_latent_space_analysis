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

    def _compute_empirical_derivatives(self, outputs: torch.Tensor, time_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the time derivative (ds_t/dt or de_t/dt) using PyTorch autograd.

        :param torch.Tensor outputs: The predicted tensor. Shape: [Batch, Seq_len, Features].
        :param torch.Tensor time_tensor: The time input tensor. Shape: [Batch, Seq_len, 1].
        :return torch.Tensor: The computed derivative tensor. Shape: [Batch, Seq_len, Features].
        """
        grads = self._get_jacobian_rows(outputs=outputs, inputs=time_tensor)
        return torch.cat(tensors=grads, dim=-1)

    def _compute_feature_gradient_trick(
            self, s_t_f: torch.Tensor, e_t_f_grad: torch.Tensor, de_dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the chain-rule gradient trick for the Feature module to compute ds_t^F/dt.
        (We assume that e_t and e_t^F are very close to each other - we achieve this using L_emb loss function)
        Formula: ds_t^F/dt = (ds_t^F/de_t^F) * (de_t/dt)

        :param torch.Tensor s_t_f: The predicted states from the Feature module.
        :param torch.Tensor e_t_f_grad: The embeddings with requires_grad=True.
        :param torch.Tensor de_dt: The empirical derivative of embeddings w.r.t time from the Time module.
        :return torch.Tensor: The empirical time derivative of the Feature module's states.
        """
        ds_de_rows = self._get_jacobian_rows(outputs=s_t_f, inputs=e_t_f_grad)
        ds_dt_list = [(de_dt * ds_i_de).sum(dim=-1, keepdim=True) for ds_i_de in ds_de_rows]
        return torch.cat(tensors=ds_dt_list, dim=-1)
