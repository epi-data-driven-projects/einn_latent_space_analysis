from einn.training.base_phase_optimizer import BasePhaseOptimizer
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext


class PhaseFourOptimizer(BasePhaseOptimizer):
    """
    Phase 4: Global Fine-Tuning and Parameter Discovery.
    Both Time and Feature modules are frozen. Only the shared Output module AND
    the physical ODE parameters (e.g., beta, gamma) are updated to find the optimal
    physical dynamics that fit the latent representations.
    """

    def prepare_network_states(self, context: PhaseContext) -> None:
        """Freezes Time and Feature modules, unfreezes Output module."""
        # Freeze Time module
        context.models.time_module.freeze_parameters()
        context.models.time_module.set_eval_mode()

        # Freeze Feature module
        context.models.feature_module.freeze_parameters()
        context.models.feature_module.set_eval_mode()

        # Unfreeze Output module
        context.models.output_module.unfreeze_parameters()
        context.models.output_module.set_train_mode()

        # Note: The ODE parameters (BaseODEModel.raw_params) are always requires_grad=True,
        # but they only get updated now because they are passed to this phase's optimizer
        # in the EINNTrainer setup.

    def step(self, context: PhaseContext) -> float:
        """
        Executes the optimization step for Phase 4.
        """
        self.prepare_network_states(context)
        self.optimizer.zero_grad()

        # Run both modules to accumulate gradients for the shared Output module
        # from both the Time and Feature pathways.
        e_t, s_t, ds_t_dt, de_t_dt = self.forward_engine.forward_time_and_ode(context.t)
        e_t_F, s_t_F, ds_t_F_dt = self.forward_engine.forward_gradient_feature(context.X, context.t, de_t_dt)

        outputs = NetworkOutputs(e_t=e_t, e_t_F=e_t_F, s_t=s_t, s_t_F=s_t_F)

        # Calculate loss using gradients from BOTH branches.
        # CRITICAL: detach_ode_params=False allows the ODE parameters to be updated!
        loss = self.loss_calculator(context, outputs, ds_dt_t=ds_t_dt, ds_dt_f=ds_t_F_dt, detach_ode_params=False)

        loss.backward()
        self.optimizer.step()

        return loss.item()
