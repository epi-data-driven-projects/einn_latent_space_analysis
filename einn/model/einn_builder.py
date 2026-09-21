from typing import Optional

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.model.feature_module import FeatureModule
from einn.model.interface.einn_models import EINNModels
from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule
from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel


class EINNBuilder:
    """
    Class responsible for instantiating and assembling the EINN model components
    with explicit architectural configurations.
    """

    @staticmethod
    def build_einn(
        model_config: EINNModelConfig,
        train_config: EINNTrainConfig,
        param_calibration: dict,
        model_type: str = "SEIRM",
        population_n: float | None = 1.0,
        seed: int = 42
    ) -> EINNModels:
        """
        Builds and returns the EINNModels container based on the provided configurations.

        :param EINNModelConfig model_config: The configuration object for neural architectures.
        :param EINNTrainConfig train_config: The configuration object for training (used for device).
        :param dict param_calibration: Initial parameters for the ODE model.
        :param str model_type: Type of ODE model to build ('SIR' or 'SEIRM').
        :param Optional[float] population_n: Total population for the SEIRM model.
        :param int seed: Seed for random operations in modules (e.g., Fourier mapping).
        :return EINNModels: A container holding the initialized neural and ODE models on the correct device.
        """
        time_module = TimeModule(
            mapping_size=model_config.time_mapping_size,
            scale=model_config.time_scale,
            out_dim=model_config.d_e,
            seed=seed
        ).to(device=train_config.device)

        feature_module = FeatureModule(
            dim_seq_in=model_config.d_x,
            rnn_out=model_config.feature_rnn_out,
            dim_out=model_config.d_e,
            n_layers=model_config.feature_n_layers,
            bidirectional=model_config.feature_bidirectional,
            dropout=model_config.feature_dropout
        ).to(device=train_config.device)

        output_module = OutputModule(
            d_e=model_config.d_e,
            d_s=model_config.d_s
        ).to(device=train_config.device)

        if model_type == "SEIRM":
            ode_model = SEIRMModel(population_n=population_n)
        elif model_type == "SIR":
            ode_model = SIRModel()
        else:
            raise ValueError(f"Unsupported ODE model type: {model_type}")

        ode_model.init_params(param_dict=param_calibration)
        ode_model.to(device=train_config.device)

        return EINNModels(
            time_module=time_module,
            feature_module=feature_module,
            output_module=output_module,
            ode_model=ode_model
        )
