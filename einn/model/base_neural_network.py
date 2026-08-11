import torch.nn as nn


class BaseNeuralNetwork(nn.Module):
    """
    Abstract base class for all neural network modules in the EINN framework.
    Provides standardized methods for freezing/unfreezing parameters and managing
    training/evaluation states safely.
    """

    def __init__(self) -> None:
        super(BaseNeuralNetwork, self).__init__()

    def freeze_parameters(self) -> None:
        """
        Freezes all learnable parameters in the network by setting requires_grad = False.
        This prevents the optimizer from updating these weights during backpropagation.

        Note: We purposely DO NOT call self.eval() here. Gradient calculation (requires_grad)
        and network behavior (Dropout/BatchNorm) are orthogonal concepts in PyTorch and
        should be controlled separately.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """
        Unfreezes all learnable parameters by setting requires_grad = True.
        This allows the optimizer to update the weights during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = True

    def set_train_mode(self) -> None:
        """
        Sets the network to training mode.
        Activates training-specific layers like Dropout and updates running
        statistics in BatchNorm layers.
        """
        self.train()

    def set_eval_mode(self) -> None:
        """
        Sets the network to evaluation (inference) mode.
        Disables Dropout and freezes running statistics in BatchNorm layers
        to ensure deterministic outputs.
        """
        self.eval()
