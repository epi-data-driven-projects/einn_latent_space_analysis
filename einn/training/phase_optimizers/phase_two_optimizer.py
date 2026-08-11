from einn.training.base_phase_optimizer import BasePhaseOptimizer
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext


class PhaseTwoOptimizer(BasePhaseOptimizer):
    """
    Phase 2: Physics-Informed Feature Training.
    Feature and Output modules learn from both the data and the physical ODE constraints.
    The ODE parameters themselves are NOT updated in this phase (detach_ode_params=True).
    """

    def prepare_network_states(self, context: PhaseContext) -> None:
        """Freezes the Time module and unfreezes the Feature and Output modules."""
        # Identical to Phase 1 setup
        context.models.time_module.freeze_parameters()
        context.models.time_module.set_eval_mode()

        context.models.feature_module.unfreeze_parameters()
        context.models.output_module.unfreeze_parameters()
        context.models.feature_module.set_train_mode()
        context.models.output_module.set_train_mode()

    def step(self, context: PhaseContext) -> float:
        """
        Executes the optimization step for Phase 2.
        """
        self.prepare_network_states(context)
        self.optimizer.zero_grad()

        # 1. Forward Time module (even though it's frozen) to get the time derivative (de_t/dt).
        # This is strictly required for the chain rule in the Feature module's ODE calculation.
        e_t, s_t, ds_t_dt, de_t_dt = self.forward_engine.forward_time_and_ode(context.t)

        # 2. Forward Feature module, calculating its own time derivative (ds_t_F_dt) using de_t_dt
        e_t_F, s_t_F, ds_t_F_dt = self.forward_engine.forward_gradient_feature(context.X, context.t, de_t_dt)

        outputs = NetworkOutputs(e_t=e_t, e_t_F=e_t_F, s_t=s_t, s_t_F=s_t_F)

        # 3. Calculate loss passing the Feature module's derivative.
        # We set detach_ode_params=True so the neural network doesn't mess up the ODE parameters yet.
        loss = self.loss_calculator(context, outputs, ds_dt_f=ds_t_F_dt, detach_ode_params=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()
