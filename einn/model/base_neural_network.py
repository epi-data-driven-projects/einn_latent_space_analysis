import torch.nn as nn


class BaseNeuralNetwork(nn.Module):
    """Minden neurális modul absztrakt ősosztálya."""

    def __init__(self):
        super(BaseNeuralNetwork, self).__init__()

    def freeze_parameters(self):
        """Lefagyasztja a hálózat összes tanulható paraméterét."""
        self.eval()  # kell?

        # TODO: ellenőrizni

        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """Kiolvasztja a hálózat összes tanulható paraméterét."""
        self.train()  # kell?
        for param in self.parameters():
            param.requires_grad = True

    def set_train_mode(self):
        """Tanítási módba kapcsolja a hálózatot (pl. Dropout, BatchNorm aktiválása)."""
        self.train()

    def set_eval_mode(self):
        """Kiértékelési módba kapcsolja a hálózatot."""
        self.eval()
