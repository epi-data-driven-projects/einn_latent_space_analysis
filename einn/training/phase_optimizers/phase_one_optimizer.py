from einn.training.base_phase_optimizer import BasePhaseOptimizer
from einn.model.interface.network_outputs import NetworkOutputs
from einn.model.interface.phase_context import PhaseContext


class PhaseOneOptimizer(BasePhaseOptimizer):
    """
    Phase 1: Pure Data-Driven Training.
    Only the Feature and Output modules are trained using the observation data (Data Loss).
    The Time module is completely frozen and set to evaluation mode.
    """

    def prepare_network_states(self, context: PhaseContext) -> None:
        """Freezes the Time module and unfreezes the Feature and Output modules."""
        # Freeze Time module
        context.models.time_module.freeze_parameters()
        context.models.time_module.set_eval_mode()

        # Unfreeze and set training mode for Feature and Output modules
        context.models.feature_module.unfreeze_parameters()
        context.models.output_module.unfreeze_parameters()
        context.models.feature_module.set_train_mode()
        context.models.output_module.set_train_mode()

    def step(self, context: PhaseContext) -> float:
        """
        Executes the optimization step for Phase 1.

        Args:
            context (PhaseContext): Contains the current batch of data (X, y, t).

        Returns:
            float: The calculated scalar loss for this step.
        """
        self.prepare_network_states(context)
        self.optimizer.zero_grad()

        # Run only the Feature module pathway
        # e_t_F shape: [Batch, Seq_len, d_e]
        e_t_F = context.models.feature_module(context.X, context.t)

        # s_t_F shape: [Batch, Seq_len, d_s]
        s_t_F = context.models.output_module(e_t_F)

        # Package the outputs. Time module outputs are None in this phase.
        outputs = NetworkOutputs(e_t=None, e_t_F=e_t_F, s_t=None, s_t_F=s_t_F)

        # Calculate loss and backpropagate
        loss = self.loss_calculator(context, outputs)
        loss.backward()
        self.optimizer.step()

        return loss.item()
