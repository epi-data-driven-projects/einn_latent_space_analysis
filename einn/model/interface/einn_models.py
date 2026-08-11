from dataclasses import dataclass

from einn.model.base_neural_network import BaseNeuralNetwork
from einn.ode.base_ode_model import BaseODEModel


@dataclass
class EINNModels:
    """
    Container for the instantiated neural and physical models. Ensures that the EINNBuilder returns
    a structured and predictable object.

    - time_module (BaseNeuralNetwork): The neural network mapping time vectors to latent embeddings.
    - feature_module (BaseNeuralNetwork): The Seq2Seq neural network mapping observations to latent embeddings.
    - output_module (BaseNeuralNetwork): The shared MLP decoding latent embeddings into physical ODE states.
    - ode_model (BaseODEModel): The physical physics model (e.g., SEIRM, SIR) calculating analytical derivatives.
    """
    time_module: BaseNeuralNetwork
    feature_module: BaseNeuralNetwork
    output_module: BaseNeuralNetwork
    ode_model: BaseODEModel
