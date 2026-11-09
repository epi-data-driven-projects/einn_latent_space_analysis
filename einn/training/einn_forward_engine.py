import torch

from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_scaling_stats import DataScalingStats
from einn.model.interface.phase_context import PhaseContext
from einn.model.interface.network_outputs import NetworkOutputs


class EINNForwardEngine:
    """
    Core execution engine for the EINN architecture.
    Responsible for routing data through the Time, Feature, and Output modules, computing empirical and
    analytical derivatives, and applying the gradient trick.
    """

    def __init__(self, train_config: EINNTrainConfig, scaling_stats: DataScalingStats):
        """
        Initializes the Forward Engine with the training configuration.

        :param EINNTrainConfig train_config: Configuration containing training hyperparameters (e.g., device).
        """
        self.config = train_config
        self.scaling_stats = scaling_stats

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
        out = NetworkOutputs()
        models = context.models

        x_input = context.x
        past_steps = x_input.shape[1]

        # 1. A nyers időt NEM követjük autograd-dal
        t_raw = context.t.clone().detach()

        # Skálázási paraméterek kinyerése
        aux_mean = self.scaling_stats.aux_mean.to(t_raw.device)
        aux_std = self.scaling_stats.aux_std.to(t_raw.device)
        t_min = self.scaling_stats.t_min.to(t_raw.device)
        t_max = self.scaling_stats.t_max.to(t_raw.device)

        # 2. TELJES MIN-MAX SKÁLÁZÁS ÉS GRADIENS KÖVETÉS A NORMALIZÁLT TÉRBEN!
        t_range = (t_max - t_min + 1e-8)
        t_scaled = ((t_raw - t_min) / t_range).requires_grad_(True)

        # Phases 1-4: Time Module forward and derivatives
        e_t_full = models.time_module(t=t_scaled)
        s_t_full = models.output_module(e=e_t_full)

        # 3. DERIVÁLÁS A NORMALIZÁLT IDŐ SZERINT (ds/dt_norm)
        ds_t_dt_norm = self._compute_empirical_derivatives(outputs=s_t_full, time_tensor=t_scaled)
        
        # 4. LÁNCSZABÁLY: Visszaskálázás a valós fizikai időre (osztás az intervallummal)
        ds_t_dt_nn_full = ds_t_dt_norm / t_range

        # Transform back to the real space
        s_t_full_real = (s_t_full * aux_std) + aux_mean
        ode_solution_full = models.ode_model.get_derivatives(states=s_t_full_real, detach_params=False)

        ds_t_dt_ode_full = ode_solution_full.ds_dt / aux_std

        # Slicing the past steps into NetworkOutputs
        out.e_t = e_t_full[:, :past_steps, :]
        out.s_t = s_t_full[:, :past_steps, :]
        out.ds_dt_T_nn = ds_t_dt_nn_full[:, :past_steps, :]
        out.ds_dt_T_ode = ds_t_dt_ode_full[:, :past_steps, :]

        if ode_solution_full.params.dim() == 3:
            out.params = ode_solution_full.params[:, :past_steps, :]
        else:
            out.params = ode_solution_full.params

        if context.phase_num >= 2:
            out.ds_dt_future_T_nn = ds_t_dt_nn_full[:, past_steps:, :]
            out.ds_dt_future_T_ode = ds_t_dt_ode_full[:, past_steps:, :]

        if context.phase_num >= 3:
            # Itt a detach fontos, hogy ne menjen vissza gradiens az időn keresztül
            e_t_f_full = models.feature_module(x=x_input, t=t_scaled.detach(), mask=context.mask)
            
            if context.phase_num == 4:
                e_t_f_full = e_t_f_full.detach()
            
            s_t_f_full = models.output_module(e=e_t_f_full)

            out.e_t_F = e_t_f_full[:, :past_steps, :]
            out.s_t_F = s_t_f_full[:, :past_steps, :]

        if context.phase_num == 4:
            # 5. UGYANEZ A LÁNCSZABÁLY A LÁTENS TÉR DERIVÁLTJÁRA IS
            de_dt_norm = self._compute_empirical_derivatives(outputs=e_t_full, time_tensor=t_scaled)
            de_dt_full = de_dt_norm / t_range

            e_t_f_grad_full = e_t_f_full.clone().detach().requires_grad_(True)
            s_t_f_grad_full = models.output_module(e=e_t_f_grad_full)

            # Apply gradient trick for the FULL sequence
            ds_dt_f_nn_full = self._compute_feature_gradient_trick(
                s_t_f=s_t_f_grad_full, e_t_f_grad=e_t_f_grad_full, de_dt=de_dt_full
            )

            s_t_f_grad_full_real = (s_t_f_grad_full * aux_std) + aux_mean
            ds_dt_f_ode_full = models.ode_model.get_derivatives(states=s_t_f_grad_full_real,
                                                                detach_params=True).ds_dt / aux_std
                                                                
            out.ds_dt_F_nn = ds_dt_f_nn_full[:, :past_steps, :]
            out.ds_dt_F_ode = ds_dt_f_ode_full[:, :past_steps, :]
            out.ds_dt_future_F_nn = ds_dt_f_nn_full[:, past_steps:, :]
            out.ds_dt_future_F_ode = ds_dt_f_ode_full[:, past_steps:, :]

        return out
