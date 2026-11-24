"""
Dataclass container bundling model-specific and shared submodules for multi-model EINN.

Groups the per-model TimeModules and mechanistic ODE models alongside the shared
FeatureModule, shared OutputModule, and model-label registry (thesis Section 2.4, Figure 9).
"""
from dataclasses import dataclass

from torch import nn

from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule
from einn.ode.base_ode_model import BaseODEModel
from einn_multimodel.model.feature_module import FeatureModule


@dataclass
class EINNModels:
    """
    Container holding all neural and mechanistic submodules of a single- or multi-model EINN.

    Enforces key consistency across time_modules, ode_models, and model_to_id, and provides
    single-model convenience properties (time_module, ode_model) for backward compatibility.

    :param nn.ModuleDict time_modules: Model-specific TimeModule instances keyed by uppercase
        model name (each mapping scaled time (B, T, 1) -> e_t^{T_m} of shape (B, T, d_e)).
    :param nn.ModuleDict ode_models: Model-specific BaseODEModel instances keyed by uppercase
        model name (each evaluating ds/dt = f_ODE(s, Omega) of shape (B, T, d_s)).
    :param FeatureModule feature_module: Shared model-label-conditioned encoder-decoder mapping
        (x, t, mask, model_label) -> e_t^F of shape (B, T, d_e).
    :param OutputModule output_module: Shared MLP mapping latent embeddings (B, T, d_e) ->
        normalized compartment states (B, T, d_s).
    :param dict[str, int] model_to_id: Mapping from each model name to its unique contiguous
        integer class ID in [0, K - 1].
    """

    time_modules: nn.ModuleDict
    ode_models: nn.ModuleDict
    feature_module: FeatureModule
    output_module: OutputModule
    model_to_id: dict[str, int]

    def __post_init__(self) -> None:
        """
        Validate that time_modules, ode_models, and model_to_id share identical non-empty keys
        and unique integer class IDs.

        :return None: Validates container invariants in place.
        :raises ValueError: If module dictionaries are empty, keys mismatch, or model IDs collide.
        """
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
        """
        Return the sole TimeModule for backward compatibility in single-model experiments.

        :return TimeModule: The single registered TimeModule instance.
        :raises AttributeError: If more than one model family is registered in time_modules.
        """
        if len(self.time_modules) != 1:
            raise AttributeError("time_module is ambiguous for a multi-model EINN; use time_modules.")
        return next(iter(self.time_modules.values()))

    @property
    def ode_model(self) -> BaseODEModel:
        """
        Return the sole BaseODEModel for backward compatibility in single-model experiments.

        :return BaseODEModel: The single registered BaseODEModel instance.
        :raises AttributeError: If more than one model family is registered in ode_models.
        """
        if len(self.ode_models) != 1:
            raise AttributeError("ode_model is ambiguous for a multi-model EINN; use ode_models.")
        return next(iter(self.ode_models.values()))
