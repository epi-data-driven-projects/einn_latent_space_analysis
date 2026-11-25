from dataclasses import dataclass

from torch import nn

from einn.model.output_module import OutputModule
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

