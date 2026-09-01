import torch
import torch.nn as nn


class BaseNeuralNetwork(nn.Module):
    """
    Abstract base class for all neural network modules in the EINN framework. Provides standardized methods for
     freezing/unfreezing parameters and managing training/evaluation states safely.
    """

    def __init__(self):
        super().__init__()

    def freeze_parameters(self):
        """
        Freezes all learnable parameters in the network by setting requires_grad = False.
        This prevents the optimizer from updating these weights during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """
        Unfreezes all learnable parameters by setting requires_grad = True.
        This allows the optimizer to update the weights during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = True

    def set_train_mode(self):
        """
        Sets the network to training mode.
        Activates training-specific layers like Dropout and updates running
        statistics in BatchNorm layers.
        """
        self.train()

    def set_eval_mode(self):
        """
        Sets the network to evaluation (inference) mode.
        Disables Dropout and freezes running statistics in BatchNorm layers
        to ensure deterministic outputs.
        """
        self.eval()

    @staticmethod
    def _init_weights(m: nn.Module):
        """
        Shared weight initialization strategy for all EINN neural networks.
        Applies Xavier uniform initialization to linear layers and fills biases with 0.01.
        :param nn.Module m: A PyTorch module (e.g., nn.Linear, nn.GRU).
        """

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
