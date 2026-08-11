import torch
from einn.training.base_phase_optimizer import BasePhaseOptimizer
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext


class PhaseThreeOptimizer(BasePhaseOptimizer):
    """
    Phase 3: Knowledge Distillation.
    The Time and Output modules are trained. The Time module learns to mimic the latent
    embeddings and physical states of the frozen Feature module (Knowledge Distillation),
    while also respecting the ODE constraints.
    """

    def prepare_network_states(self, context: PhaseContext) -> None:
        """Freezes the Feature module and unfreezes the Time and Output modules."""
        # Freeze Feature module
        context.models.feature_module.freeze_parameters()
        context.models.feature_module.set_eval_mode()

        # Unfreeze and set training mode for Time and Output modules
        context.models.time_module.unfreeze_parameters()
        context.models.output_module.unfreeze_parameters()
        context.models.time_module.set_train_mode()
        context.models.output_module.set_train_mode()

    def step(self, context: PhaseContext) -> float:
        """
        Executes the optimization step for Phase 3.
        """
        self.prepare_network_states(context)
        self.optimizer.zero_grad()

        # 1. Run the Feature module without tracking gradients to act as the "Teacher" target
        with torch.no_grad():
            e_t_F = context.models.feature_module(context.X, context.t)
            s_t_F = context.models.output_module(e_t_F)

        # 2. Run the Time module (with gradients) acting as the "Student"
        e_t, s_t, ds_t_dt, de_t_dt = self.forward_engine.forward_time_and_ode(context.t)

        outputs = NetworkOutputs(e_t=e_t, e_t_F=e_t_F, s_t=s_t, s_t_F=s_t_F)

        # 3. Calculate loss passing the Time module's derivative.
        # ODE parameters remain frozen (detached) in this phase.
        loss = self.loss_calculator(context, outputs, ds_dt_t=ds_t_dt, detach_ode_params=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()
