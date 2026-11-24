"""
Factory for assembling single-model and two-model (SIR/SIS, SIR/SEIR) EINN architectures.

Instantiates model-specific TimeModules and mechanistic ODE models alongside the shared
FeatureModule and OutputModule described in thesis Section 2.4 (Figure 9).
"""
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule
from einn.ode.seir_model import SEIRModel
from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel
from einn_multimodel.config.einn_model_config import EINNModelConfig
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.model.feature_module import FeatureModule
from einn_multimodel.model.interface.einn_models import EINNModels
from einn_multimodel.model.model_labels import validate_model_registry
from einn_multimodel.ode.sir_seir_compatible_model import SIRSEIRCompatibleModel
from einn_multimodel.ode.sis_model import SISModel


class EINNBuilder:
    """
    Builder for single-model and two-model (SIR/SIS, SIR/SEIR) EINN module containers.

    Validates cross-model dimension compatibility (d_x, d_s, d_p, num_models) and places all
    constructed neural and mechanistic submodules on the configured training device.
    """

    @staticmethod
    def _build_shared_modules(
            model_config: EINNModelConfig,
            model_names: list[str],
            device: str,
            seed: int = 42,
    ) -> tuple[nn.ModuleDict, FeatureModule, OutputModule, dict[str, int]]:
        """
        Instantiate per-model TimeModules, the shared FeatureModule, and the shared OutputModule.

        Each TimeModule receives an offset Fourier-feature seed (seed + idx) so paired models
        start from independent random frequency bases b_gauss in R^m (thesis Section 2.2/2.4).

        :param EINNModelConfig model_config: Architectural hyperparameters (d_x, d_e, d_s,
            RNN settings, num_models, model_label_dim, and conditioning flags).
        :param list[str] model_names: Ordered list of uppercase epidemiological model identifiers.
        :param str device: Target PyTorch device string (e.g. cpu or cuda:0).
        :param int seed: Base random seed for TimeModule Fourier feature initialization.
        :return tuple[nn.ModuleDict, FeatureModule, OutputModule, dict[str, int]]: Tuple
            (time_modules, feature_module, output_module, model_to_id) moved to device.
        :raises ValueError: If model_config.num_models does not match len(model_names).
        """
        model_to_id = validate_model_registry(model_names)
        if model_config.num_models != len(model_names):
            raise ValueError(
                f"model_config.num_models={model_config.num_models}, "
                f"but {len(model_names)} models were requested."
            )

        # Model-specific TimeModules mapping scaled time (B, T, 1) -> e_t^{T_m} (B, T, d_e).
        time_modules = nn.ModuleDict({
            name: TimeModule(
                mapping_size=model_config.time_mapping_size,
                scale=model_config.time_scale,
                out_dim=model_config.d_e,
                seed=seed + idx,
            )
            for idx, name in enumerate(model_names)
        }).to(device)

        # Shared FeatureModule mapping (X (B, T_obs, d_x), t (B, T, 1), label (B,)) -> e_t^F (B, T, d_e).
        feature_module = FeatureModule(
            dim_seq_in=model_config.d_x,
            rnn_out=model_config.feature_rnn_out,
            dim_out=model_config.d_e,
            n_layers=model_config.feature_n_layers,
            bidirectional=model_config.feature_bidirectional,
            dropout=model_config.feature_dropout,
            num_models=len(model_names),
            model_label_dim=model_config.model_label_dim,
            normalize_embeddings=model_config.normalize_feature_embeddings,
            condition_on_model_label=model_config.condition_feature_on_true_label,
        ).to(device)

        # Shared OutputModule mapping latent embeddings (B, T, d_e) -> compartment states (B, T, d_s).
        output_module = OutputModule(
            d_e=model_config.d_e,
            d_s=model_config.d_s,
        ).to(device)
        return time_modules, feature_module, output_module, model_to_id

    @staticmethod
    def build_multi_einn(
            model_config: EINNModelConfig,
            train_config: EINNTrainConfig,
            param_calibration: Mapping[str, dict[str, Any]],
            population_n: float = 1.0,
            seed: int = 42,
            model_names: Sequence[str] = ("SIR", "SIS"),
    ) -> EINNModels:
        """
        Assemble a validated two-model EINN for either the SIR/SIS or SIR/SEIR model pair.

        Enforces the unified compartment layouts from thesis Section 2.4:
        SIR/SIS uses d_s = 3, d_p = 2, target_idx = 1;
        SIR/SEIR uses d_s = 4, d_p = 3, target_idx = 2 (with SIRSEIRCompatibleModel padding E = 0).

        :param EINNModelConfig model_config: Neural network and unified ODE dimension config.
        :param EINNTrainConfig train_config: Training config providing the target compute device.
        :param Mapping[str, dict[str, Any]] param_calibration: Mapping from each model name in
            model_names to its initial mechanistic parameter dictionary (e.g. {'beta': ..., 'gamma': ...}).
        :param float population_n: Total population N (1.0 for normalized population fractions).
        :param int seed: Base random seed for TimeModule Fourier feature sampling.
        :param Sequence[str] model_names: Two distinct model names forming a supported pair.
        :return EINNModels: Assembled multi-model container ready for training or inference.
        :raises ValueError: If model_names is unsupported, dimensions (d_s, d_p, d_x, num_models)
            conflict with the pair layout, or param_calibration keys do not match model_names.
        """
        names = [name.upper() for name in model_names]
        pair = frozenset(names)
        if pair not in {
            frozenset(("SIR", "SIS")),
            frozenset(("SIR", "SEIR")),
        }:
            raise ValueError("Supported model pairs are SIR/SIS and SIR/SEIR.")
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError("Exactly two distinct model names are required.")

        # Validate unified state, parameter, and synthetic input dimensions for the chosen pair.
        expected_d_s = 3 if pair == frozenset(("SIR", "SIS")) else 4
        expected_d_p = 2 if pair == frozenset(("SIR", "SIS")) else 3
        if model_config.d_s != expected_d_s:
            raise ValueError(
                f"{names} requires shared d_s={expected_d_s}, got {model_config.d_s}."
            )
        if model_config.d_p != expected_d_p:
            raise ValueError(
                f"{names} requires shared d_p={expected_d_p}, got {model_config.d_p}."
            )
        if model_config.d_x != 5:
            raise ValueError(
                "SyntheticDataGenerator produces five input features "
                "(four covariates plus noisy I); set d_x=5."
            )
        if model_config.num_models != 2:
            raise ValueError("A two-model experiment requires num_models=2.")
        if set(param_calibration) != set(names):
            raise ValueError(
                f"param_calibration must contain exactly {sorted(names)}."
            )

        time_modules, feature_module, output_module, model_to_id = (
            EINNBuilder._build_shared_modules(
                model_config=model_config,
                model_names=names,
                device=train_config.device,
                seed=seed,
            )
        )

        # Select pair-compatible ODE factories sharing the same target_idx and state dimension d_s.
        if pair == frozenset(("SIR", "SIS")):
            factories = {
                "SIR": lambda: SIRModel(population_n=population_n, target_idx=1),
                "SIS": lambda: SISModel(population_n=population_n, target_idx=1),
            }
        else:
            factories = {
                "SIR": lambda: SIRSEIRCompatibleModel(
                    population_n=population_n, target_idx=2
                ),
                "SEIR": lambda: SEIRModel(
                    population_n=population_n, target_idx=2
                ),
            }

        # Instantiate and calibrate each model's learnable ODE parameters on the target device.
        ode_models = nn.ModuleDict()
        for name in names:
            model = factories[name]()
            model.init_params(param_dict=param_calibration[name])
            ode_models[name] = model.to(train_config.device)

        return EINNModels(
            time_modules=time_modules,
            ode_models=ode_models,
            feature_module=feature_module,
            output_module=output_module,
            model_to_id=model_to_id,
        )

    @staticmethod
    def build_einn(
            model_config: EINNModelConfig,
            train_config: EINNTrainConfig,
            param_calibration: dict[str, Any],
            model_type: str = "SEIRM",
            population_n: float | None = 1.0,
            seed: int = 42,
    ) -> EINNModels:
        """
        Backward-compatible builder wrapping a single epidemiological model in EINNModels.

        :param EINNModelConfig model_config: Neural network and ODE dimension configuration.
        :param EINNTrainConfig train_config: Training configuration supplying the target device.
        :param dict[str, Any] param_calibration: Initial physical parameter dictionary for the ODE model.
        :param str model_type: Epidemiological model identifier (SIR, SIS, SEIR, or SEIRM).
        :param float | None population_n: Total population N used by SEIRM (defaults to 1.0 if None).
        :param int seed: Random seed for TimeModule Fourier feature initialization.
        :return EINNModels: Single-entry EINNModels container with model_to_id={model_type: 0}.
        :raises ValueError: If model_type is not in {SIR, SIS, SEIR, SEIRM}.
        """
        model_type = model_type.upper()
        factories = {
            "SIR": SIRModel,
            "SIS": SISModel,
            "SEIR": SEIRModel,
            "SEIRM": lambda: SEIRMModel(population_n=population_n or 1.0),
        }
        if model_type not in factories:
            raise ValueError(f"Unsupported ODE model type: {model_type}")

        time_module = TimeModule(
            mapping_size=model_config.time_mapping_size,
            scale=model_config.time_scale,
            out_dim=model_config.d_e,
            seed=seed,
        ).to(train_config.device)
        feature_module = FeatureModule(
            dim_seq_in=model_config.d_x,
            rnn_out=model_config.feature_rnn_out,
            dim_out=model_config.d_e,
            n_layers=model_config.feature_n_layers,
            bidirectional=model_config.feature_bidirectional,
            dropout=model_config.feature_dropout,
            num_models=1,
            model_label_dim=model_config.model_label_dim,
            normalize_embeddings=model_config.normalize_feature_embeddings,
            condition_on_model_label=model_config.condition_feature_on_true_label,
        ).to(train_config.device)
        output_module = OutputModule(d_e=model_config.d_e, d_s=model_config.d_s).to(
            train_config.device
        )

        ode_model = factories[model_type]()
        ode_model.init_params(param_dict=param_calibration)
        ode_model.to(train_config.device)

        return EINNModels(
            time_modules=nn.ModuleDict({model_type: time_module}),
            ode_models=nn.ModuleDict({model_type: ode_model}),
            feature_module=feature_module,
            output_module=output_module,
            model_to_id={model_type: 0},
        )
