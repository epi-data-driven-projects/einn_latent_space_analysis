from dataclasses import dataclass

from torch import nn

from einn_multimodel.model.feature_module import FeatureModule
from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule
from einn.ode.base_ode_model import BaseODEModel


@dataclass
class EINNModels:
    """Container for model-specific Time/ODE modules and shared neural modules."""

    time_modules: nn.ModuleDict
    ode_models: nn.ModuleDict
    feature_module: FeatureModule
    output_module: OutputModule
    model_to_id: dict[str, int]

    def __post_init__(self) -> None:
        time_names = set(self.time_modules.keys())
        ode_names = set(self.ode_models.keys())
        if not time_names or time_names != ode_names:
            raise ValueError("time_modules and ode_models must have identical non-empty keys.")
        if set(self.model_to_id) != time_names:
            raise ValueError("model_to_id must contain exactly the model names in ModuleDicts.")
        if len(set(self.model_to_id.values())) != len(self.model_to_id):
            raise ValueError("Model ids must be unique.")

    @property
    def time_module(self) -> TimeModule:
        """Backward-compatible accessor for single-model experiments only."""
        if len(self.time_modules) != 1:
            raise AttributeError("time_module is ambiguous for a multi-model EINN; use time_modules.")
        return next(iter(self.time_modules.values()))

    @property
    def ode_model(self) -> BaseODEModel:
        """Backward-compatible accessor for single-model experiments only."""
        if len(self.ode_models) != 1:
            raise AttributeError("ode_model is ambiguous for a multi-model EINN; use ode_models.")
        return next(iter(self.ode_models.values()))
