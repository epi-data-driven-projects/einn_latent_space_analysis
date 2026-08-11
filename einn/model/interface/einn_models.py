from dataclasses import dataclass

from einn.model.base_neural_network import BaseNeuralNetwork
from einn.ode.base_ode_model import BaseODEModel


@dataclass
class EINNModels:
    """
    Type-safe container for the instantiated neural and physical models.
    Ensures that the EINNBuilder returns a structured and predictable object
    rather than a generic dictionary.
    """
    # TODO: param docstrings?
    time_module: BaseNeuralNetwork
    feature_module: BaseNeuralNetwork
    output_module: BaseNeuralNetwork
    ode_model: BaseODEModel
