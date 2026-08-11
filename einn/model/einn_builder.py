from typing import Optional, Dict, Any

import torch

from einn.config.einn_config import EINNConfig
from einn.model.interface.einn_models import EINNModels
from einn.model.feature_module import FeatureModule
from einn.model.output_module import OutputModule
from einn.ode.seirm_model import SEIRMModel
from einn.model.time_module import TimeModule


class EINNBuilder:
    """
    Factory class responsible for instantiating, initializing, and moving
    all neural and physical models to the appropriate compute device (GPU/CPU).
    """

    @staticmethod
    def build_einn(config: EINNConfig, seq_len: int, param_calibration: Optional[Dict[str, Any]] = None) -> EINNModels:
        """
        Builds the entire EINN architecture based on the configuration.

        Args:
            config (EINNConfig): Configuration object containing hyperparameters and device info.
            seq_len (int): Length of the time series sequence (needed for time-dependent ODE parameters).
            param_calibration (Optional[Dict[str, Any]]): Pre-calibrated parameters for the ODE model.

        Returns:
            EINNModels: A dataclass containing the initialized models, already moved to the target device.
        """
        device = torch.device(config.device)

        # Instantiate TimeModule mapping time to embeddings
        time_module = TimeModule(
            mapping_size=config.d_e,
            scale=1.0,
            out_dim=config.d_e
        ).to(device)

        # Instantiate FeatureModule acting as Seq2Seq encoder-decoder
        feature_module = FeatureModule(
            dim_seq_in=config.d_x,
            rnn_out=40,
            dim_out=config.d_e,
            n_layers=1,
            bidirectional=True,
            dropout=0.0
        ).to(device)

        # Instantiate OutputModule decoding embeddings to physical states
        output_module = OutputModule(
            d_e=config.d_e,
            d_s=config.d_s
        ).to(device)

        # Instantiate the ODE model and initialize parameters if provided
        ode_model = SEIRMModel(
            seq_len=seq_len,
            pop_total=1.0
        ).to(device)

        if param_calibration:
            ode_model.init_params(param_dict=param_calibration)

        return EINNModels(
            time_module=time_module,
            feature_module=feature_module,
            output_module=output_module,
            ode_model=ode_model
        )
