from dataclasses import dataclass

from einn.ode.base_ode_model import BaseODEModel
from einn.model.feature_module import FeatureModule
from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule


@dataclass
class EINNModels:
    """
    Container data class for the instantiated neural networks and ODE model.

    :param TimeModule time_module: the initialized TimeModule
    :param FeatureModule feature_module: the initialized FeatureModule
    :param OutputModule output_module: the initialized shared OutputModule
    :param BaseODEModel ode_model: the initialized epidemiological ODE model
    """
    time_module: TimeModule
    feature_module: FeatureModule
    output_module: OutputModule
    ode_model: BaseODEModel
