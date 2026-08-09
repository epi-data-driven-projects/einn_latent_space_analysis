from dataclasses import dataclass
from einn.model.base_neural_network import BaseNeuralNetwork
from einn.ode.base_ode_model import BaseODEModel


@dataclass
class EINNModels:
    """A legyártott modellek típusbiztos konténere."""
    time_module: BaseNeuralNetwork
    feature_module: BaseNeuralNetwork
    output_module: BaseNeuralNetwork
    ode_model: BaseODEModel
