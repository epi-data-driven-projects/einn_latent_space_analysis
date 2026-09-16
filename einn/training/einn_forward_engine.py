import torch

from einn.config.einn_train_config import EINNTrainConfig
from einn.model.interface.phase_context import PhaseContext
from einn.model.interface.network_outputs import NetworkOutputs


class EINNForwardEngine:
    """
    Core execution engine for the EINN architecture.
    Responsible for routing data through the Time, Feature, and Output modules, computing empirical and
    analytical derivatives, and applying the gradient trick.
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

    def forward(self, context: PhaseContext) -> NetworkOutputs:
        """
        Executes the forward pass based on the current optimization phase.

        :param PhaseContext context: Contains the models, inputs, and phase identifier.
        :return NetworkOutputs: Dataclass containing all required tensors for the active phase.
        """
        out = NetworkOutputs()
        models = context.models

        x_input = context.X
        past_steps = x_input.shape[1]

        # t_grad covers the ENTIRE sequence (past + future steps)
        t_grad = context.t.clone().detach().requires_grad_(True)

        # Phases 1-4: Time Module Forward and Derivatives
        e_t_full = models.time_module(t=t_grad)
        raw_s_t_full = models.output_module(e=e_t_full)
        s_t_full = torch.sigmoid(input=raw_s_t_full)

        ds_dt_t_nn_full = self._compute_empirical_derivatives(outputs=s_t_full, time_tensor=t_grad)
        ode_solution_full = models.ode_model.get_derivatives(states=s_t_full, detach_params=False)
        ds_dt_t_ode_full = ode_solution_full.ds_dt

        # Slicing the past steps into NetworkOutputs
        out.e_t = e_t_full[:, :past_steps, :]
        out.s_t = s_t_full[:, :past_steps, :]
        out.ds_dt_T_nn = ds_dt_t_nn_full[:, :past_steps, :]
        out.ds_dt_T_ode = ds_dt_t_ode_full[:, :past_steps, :]

        # If the ODE model uses global, time-independent parameters (e.g., standard SIR/SEIRM),
        # `params` is a 1D tensor [d_p]. It cannot and should not be sliced along the time axis.
        # If the model dynamically predicts time-varying parameters (e.g., beta(t) via a neural net),
        # `params` becomes a 3D tensor [Batch, Seq_len, d_p], which MUST be sliced.
        if ode_solution_full.params.dim() == 3:
            out.params = ode_solution_full.params[:, :past_steps, :]
        else:
            out.params = ode_solution_full.params

        # Phases 2-4: Future Time Module Derivatives
        if context.phase_num >= 2:
            # Slicing the future steps into NetworkOutputs
            out.ds_dt_future_T_nn = ds_dt_t_nn_full[:, past_steps:, :]
            out.ds_dt_future_T_ode = ds_dt_t_ode_full[:, past_steps:, :]

        # Phases 3-4: Feature Module Forward
        if context.phase_num >= 3:
            # Feature module requires x, t, and mask
            # It inherently decodes the entire sequence (past + future) provided in t_grad
            e_t_f_full = models.feature_module(x=x_input, t=t_grad, mask=None)
            raw_s_t_f_full = models.output_module(e=e_t_f_full)
            s_t_f_full = torch.sigmoid(input=raw_s_t_f_full)

            # Slicing the past steps for the feature module targets
            out.e_t_F = e_t_f_full[:, :past_steps, :]
            out.s_t_F = s_t_f_full[:, :past_steps, :]

        # Phase 4
        if context.phase_num == 4:
            de_dt_full = self._compute_empirical_derivatives(outputs=e_t_full, time_tensor=t_grad)

            e_t_f_grad_full = e_t_f_full.clone().detach().requires_grad_(True)
            s_t_f_grad_full = torch.sigmoid(input=models.output_module(e=e_t_f_grad_full))

            # Apply gradient trick for the FULL sequence
            ds_dt_f_nn_full = self._compute_feature_gradient_trick(
                s_t_f=s_t_f_grad_full, e_t_f_grad=e_t_f_grad_full, de_dt=de_dt_full
            )
            ds_dt_f_ode_full = models.ode_model.get_derivatives(states=s_t_f_grad_full, detach_params=True).ds_dt

            # Populating current steps
            out.ds_dt_F_nn = ds_dt_f_nn_full[:, :past_steps, :]
            out.ds_dt_F_ode = ds_dt_f_ode_full[:, :past_steps, :]

            # Populating future steps
            out.ds_dt_future_F_nn = ds_dt_f_nn_full[:, past_steps:, :]
            out.ds_dt_future_F_ode = ds_dt_f_ode_full[:, past_steps:, :]

        return out
