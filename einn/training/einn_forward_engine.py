import torch

from einn.model.interface.einn_models import EINNModels


class EINNForwardEngine:
    """
    Engine responsible for handling the complex forward passes and computing
    gradients (time derivatives) using PyTorch's autograd system.
    This encapsulates the core Physics-Informed Neural Network (PINN) math.
    """

    def __init__(self, models: EINNModels) -> None:
        """
        Initializes the forward engine.

        Args:
            models (EINNModels): Type-safe container with the initialized networks.
        """
        self.models = models

    def forward_time_and_ode(self, t: torch.Tensor):
        """
        Executes the Time Module pathway and calculates analytical time derivatives.

        Args:
            t (torch.Tensor): Scaled time vector. Shape: [Batch, Seq_len, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - e_t: Latent time embedding [Batch, Seq_len, d_e]
                - s_t: Physical ODE states [Batch, Seq_len, d_s]
                - ds_t_dt: State time derivative (ds/dt) [Batch, Seq_len, d_s]
                - de_t_dt: Embedding time derivative (de/dt) [Batch, Seq_len, d_e]
        """
        # Enable gradient tracking for the input time tensor to allow autograd
        t.requires_grad_(True)

        # 1. Time Module forward pass
        e_t = self.models.time_module(t)

        # 2. Output Module forward pass mapping embeddings to physical states
        s_t = self.models.output_module(e_t)

        # 3. Calculate gradients (derivatives with respect to time) using Autograd.
        # create_graph=True is CRITICAL for PINNs to allow backpropagation through the derivative itself!

        # de_t / dt
        de_t_dt = torch.autograd.grad(
            outputs=e_t,
            inputs=t,
            grad_outputs=torch.ones_like(e_t),
            create_graph=True,
            retain_graph=True
        )[0]

        # ds_t / dt
        ds_t_dt = torch.autograd.grad(
            outputs=s_t,
            inputs=t,
            grad_outputs=torch.ones_like(s_t),
            create_graph=True
        )[0]

        return e_t, s_t, ds_t_dt, de_t_dt

    def forward_gradient_feature(self, x: torch.Tensor, t: torch.Tensor, de_t_dt: torch.Tensor):
        """
        Executes the Feature Module pathway and calculates its state derivative
        using the chain rule and the time derivative from the Time Module.

        Args:
            x (torch.Tensor): Observed context features. Shape: [Batch, Seq_len, d_x].
            t (torch.Tensor): Scaled time vector. Shape: [Batch, Seq_len, 1].
            de_t_dt (torch.Tensor): Time embedding derivative from Time Module. Shape: [Batch, Seq_len, d_e].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - e_t_F: Latent feature embedding [Batch, Seq_len, d_e]
                - s_t_F: Physical ODE states [Batch, Seq_len, d_s]
                - ds_t_F_dt: State time derivative (ds/dt) [Batch, Seq_len, d_s]
        """
        # 1. Feature Module forward pass
        e_t_F = self.models.feature_module(x, t)

        # Enable gradient tracking on the feature embedding to compute the Jacobian
        e_t_F_with_grad = e_t_F.clone().detach().requires_grad_(True)

        # 2. Output Module forward pass
        s_t_F = self.models.output_module(e_t_F_with_grad)

        # 3. Compute Jacobian matrix / gradient: ds_t^F / de_t^F
        ds_t_F_de_t_F = torch.autograd.grad(
            outputs=s_t_F,
            inputs=e_t_F_with_grad,
            grad_outputs=torch.ones_like(s_t_F),
            create_graph=True
        )[0]

        # 4. Apply Chain Rule: ds_t^F / dt = (ds_t^F / de_t^F) * (de_t / dt)
        # We detach de_t_dt because gradients should not flow back into the Time module here.
        ds_t_F_dt = ds_t_F_de_t_F * de_t_dt.detach()

        return e_t_F, s_t_F, ds_t_F_dt
