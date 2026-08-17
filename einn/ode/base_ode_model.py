from abc import ABC

import torch.nn as nn


class BaseODEModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        pass
