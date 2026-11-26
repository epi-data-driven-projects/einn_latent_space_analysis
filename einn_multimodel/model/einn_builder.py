from typing import Any, Mapping, Sequence

import torch.nn as nn

from einn.model.output_module import OutputModule
from einn.model.time_module import TimeModule
from einn.ode.seir_model import SEIRModel
from einn.ode.seirm_model import SEIRMModel
from einn.ode.sir_model import SIRModel
from einn.ode.sis_model import SISModel
from einn_multimodel.config.einn_model_config import EINNModelConfig
from einn_multimodel.config.einn_train_config import EINNTrainConfig
from einn_multimodel.model.feature_module import FeatureModule
from einn_multimodel.model.interface.einn_models import EINNModels
from einn_multimodel.ode.sir_seir_compatible_model import SIRSEIRCompatibleModel
from einn_multimodel.ode.sir_sis_compatible_model import SIRSISCompatibleModel


class EINNBuilder:
    """
    Builder for single-model and two-model EINN module containers.

    The single-model path uses the ODE implementations in ``einn.ode``.
    The multi-model path uses pair-compatible ODE implementations where
    necessary to give both models a shared state layout.
    """

    @staticmethod
    def validate_model_registry(model_names: Sequence[str]) -> dict[str, int]:
        """
        Normalize model names and create their integer label mapping.

        :param Sequence[str] model_names: Ordered model names.
        :return dict[str, int]: Mapping from normalized model names to label IDs.
        :raises ValueError: If the sequence is empty, contains invalid names,
            or contains duplicates after normalization.
        """
        if not model_names:
            raise ValueError("At least one model name is required.")

        names: list[str] = []
        for name in model_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Model names must be non-empty strings.")
            names.append(name.strip().upper())

        if len(set(names)) != len(names):
            raise ValueError(
                "Model names must be unique after converting to uppercase."
            )

        return {name: idx for idx, name in enumerate(names)}

    @staticmethod
    def _build_shared_modules(
        model_config: EINNModelConfig,
        model_names: Sequence[str],
        device: str,
        seed: int = 42,
    ) -> tuple[nn.ModuleDict, FeatureModule, OutputModule, dict[str, int]]:
        """
        Build per-model TimeModules and shared Feature and Output modules.

        :param EINNModelConfig model_config: Neural architecture configuration.
        :param Sequence[str] model_names: Ordered uppercase model identifiers.
        :param str device: Target PyTorch device.
        :param int seed: Base seed used to initialize TimeModule Fourier features.
        :return tuple[nn.ModuleDict, FeatureModule, OutputModule, dict[str, int]]:
            Time modules, shared feature module, output module, and model labels.
        :raises ValueError: If the number of configured models does not match
            the number of requested model names.
        """
        names = [name.strip().upper() for name in model_names]
        model_to_id = EINNBuilder.validate_model_registry(names)

        if model_config.num_models != len(names):
            raise ValueError(
                f"model_config.num_models={model_config.num_models}, "
                f"but {len(names)} models were requested."
            )

        time_modules = nn.ModuleDict(
            {
                name: TimeModule(
                    mapping_size=model_config.time_mapping_size,
                    scale=model_config.time_scale,
                    out_dim=model_config.d_e,
                    seed=seed + idx,
                )
                for idx, name in enumerate(names)
            }
        ).to(device)

        feature_module = FeatureModule(
            dim_seq_in=model_config.d_x,
            rnn_out=model_config.feature_rnn_out,
            dim_out=model_config.d_e,
            n_layers=model_config.feature_n_layers,
            bidirectional=model_config.feature_bidirectional,
            dropout=model_config.feature_dropout,
            num_models=len(names),
            model_label_dim=model_config.model_label_dim,
            normalize_embeddings=model_config.normalize_feature_embeddings,
            condition_on_model_label=model_config.condition_feature_on_true_label,
        ).to(device)

        output_module = OutputModule(
            d_e=model_config.d_e,
            d_s=model_config.d_s,
        ).to(device)

        return time_modules, feature_module, output_module, model_to_id

    @staticmethod
    def build_multi_einn(
        model_config: EINNModelConfig,
        train_config: EINNTrainConfig,
        param_calibration: Mapping[str, Mapping[str, Any]],
        population_n: float = 1.0,
        seed: int = 42,
        model_names: Sequence[str] = ("SIR", "SIS"),
    ) -> EINNModels:
        """
        Assemble a pair-compatible two-model EINN for SIR/SIS or SIR/SEIR.

        The SIR/SIS pair uses three shared state dimensions. Its SIS member
        is ``SIRSISCompatibleModel``, while the single-model builder uses the
        standalone ``einn.ode.SISModel``.

        :param EINNModelConfig model_config: Neural and shared ODE dimensions.
        :param EINNTrainConfig train_config: Training configuration.
        :param Mapping[str, Mapping[str, Any]] param_calibration: Initial
            ODE parameters keyed by model name.
        :param float population_n: Population size used by the ODE models.
        :param int seed: Base seed for TimeModule initialization.
        :param Sequence[str] model_names: The two model names in label order.
        :return EINNModels: Initialized multi-model container.
        :raises ValueError: If the pair, dimensions, or calibration keys are invalid.
        """
        names = [name.strip().upper() for name in model_names]
        model_to_id = EINNBuilder.validate_model_registry(names)

        if len(names) != 2:
            raise ValueError("Exactly two model names are required.")

        pair = frozenset(names)
        sir_sis_pair = frozenset(("SIR", "SIS"))
        sir_seir_pair = frozenset(("SIR", "SEIR"))

        if pair not in {sir_sis_pair, sir_seir_pair}:
            raise ValueError("Supported model pairs are SIR/SIS and SIR/SEIR.")

        expected_d_s, expected_d_p = (
            (3, 2) if pair == sir_sis_pair else (4, 3)
        )

        if model_config.d_s != expected_d_s:
            raise ValueError(
                f"{names} requires shared d_s={expected_d_s}, "
                f"got {model_config.d_s}."
            )
        if model_config.d_p != expected_d_p:
            raise ValueError(
                f"{names} requires shared d_p={expected_d_p}, "
                f"got {model_config.d_p}."
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

        (
            time_modules,
            feature_module,
            output_module,
            _,
        ) = EINNBuilder._build_shared_modules(
            model_config=model_config,
            model_names=names,
            device=train_config.device,
            seed=seed,
        )

        if pair == sir_sis_pair:
            factories = {
                "SIR": lambda: SIRModel(
                    population_n=population_n,
                    target_idx=1,
                ),
                "SIS": lambda: SIRSISCompatibleModel(
                    population_n=population_n,
                    target_idx=1,
                ),
            }
        else:
            factories = {
                "SIR": lambda: SIRSEIRCompatibleModel(
                    population_n=population_n,
                    target_idx=2,
                ),
                "SEIR": lambda: SEIRModel(
                    population_n=population_n,
                    target_idx=2,
                ),
            }

        ode_models = nn.ModuleDict()
        for name in names:
            model = factories[name]()
            model.init_params(param_dict=dict(param_calibration[name]))
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
        param_calibration: Mapping[str, Any],
        model_type: str = "SEIRM",
        population_n: float | None = 1.0,
        seed: int = 42,
        target_idx: int | None = None,
    ) -> EINNModels:
        """
        Build a single-model EINN using the standalone ODE implementation.

        For SIS, this selects `einn.ode.SISModel`. It does not use the
        pair-compatible ``SIRSISCompatibleModel`` from ``einn_multimodel``.

        :param EINNModelConfig model_config: Neural network and ODE dimensions.
        :param EINNTrainConfig train_config: Training configuration.
        :param Mapping[str, Any] param_calibration: Initial ODE parameters.
        :param str model_type: ODE model name: SIR, SIS, SEIR, or SEIRM.
        :param float | None population_n: Total population size.
        :param int seed: Seed for TimeModule Fourier feature initialization.
        :param int | None target_idx: Target state index. If omitted, the
            model-specific default is used.
        :return EINNModels: Single-model container.
        """
        normalized_model_type = model_type.strip().upper()
        population = 1.0 if population_n is None else float(population_n)

        default_target_indices = {
            "SIR": 1,
            "SIS": 1,
            "SEIR": 2,
            "SEIRM": 2,
        }

        factories = {
            "SIR": lambda selected_target: SIRModel(
                population_n=population,
                target_idx=selected_target,
            ),
            "SIS": lambda selected_target: SISModel(
                population_n=population,
                target_idx=selected_target,
            ),
            "SEIR": lambda selected_target: SEIRModel(
                population_n=population,
                target_idx=selected_target,
            ),
            "SEIRM": lambda selected_target: SEIRMModel(
                population_n=population,
                target_idx=selected_target,
            ),
        }

        if normalized_model_type not in factories:
            supported = ", ".join(factories)
            raise ValueError(
                f"Unsupported ODE model type {model_type!r}. "
                f"Supported types are: {supported}."
            )

        selected_target_idx = (
            default_target_indices[normalized_model_type]
            if target_idx is None
            else target_idx
        )

        ode_model = factories[normalized_model_type](selected_target_idx)

        if model_config.d_s != ode_model.d_s:
            raise ValueError(
                f"{normalized_model_type} has d_s={ode_model.d_s}, "
                f"but model_config.d_s={model_config.d_s}."
            )
        if not 0 <= selected_target_idx < ode_model.d_s:
            raise ValueError(
                f"target_idx={selected_target_idx} is outside the valid range "
                f"for {normalized_model_type} with d_s={ode_model.d_s}."
            )

        ode_model.init_params(param_dict=dict(param_calibration))
        ode_model = ode_model.to(train_config.device)

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

        output_module = OutputModule(
            d_e=model_config.d_e,
            d_s=model_config.d_s,
        ).to(train_config.device)

        model_to_id = EINNBuilder.validate_model_registry(
            [normalized_model_type]
        )

        return EINNModels(
            time_modules=nn.ModuleDict(
                {normalized_model_type: time_module}
            ),
            ode_models=nn.ModuleDict(
                {normalized_model_type: ode_model}
            ),
            feature_module=feature_module,
            output_module=output_module,
            model_to_id=model_to_id,
        )
