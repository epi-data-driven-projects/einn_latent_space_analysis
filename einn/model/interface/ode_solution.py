from dataclasses import dataclass
import torch


@dataclass
class ODESolution:
    ds_dt: torch.Tensor
    params: torch.Tensor
    t: torch.Tensor
# TODO: add docstring and comment
