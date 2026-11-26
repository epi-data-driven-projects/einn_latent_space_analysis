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
    Builder for single-model and multi-model EINN containers.

    The multi-model path uses ODE implementations from einn_multimodel. Those
    implementations give the models in a pair a shared state layout.

    The single-model path is retained for compatibility. It uses the original
    ODE implementations from einn.ode, including the standalone SIS model.
    """

    @staticmethod
    def validate_model_registry(
        model_names: Sequence[str],
    ) -> dict[str, int]:
        """
        Normalize model names and assign each one an integer label.

        The order in model_names determines the assigned labels. For example,
        ["SIR", "SIS"] produces {"SIR": 0, "SIS": 1}. Normalizing names here
        ensures that "sir" and "SIR" cannot accidentally become separate labels.

        :param Sequence[str] model_names: Ordered model names.
        :return dict[str, int]: Mapping from normalized names to integer labels.
        """
        if not model_names:
            raise ValueError("At least one model name is required.")

        normalized_names: list[str] = []

        for name in model_names:
            # Reject invalid values before calling string methods on them.
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Model names must be non-empty strings.")

            # Store only the canonical uppercase form throughout the builder.
            normalized_names.append(name.strip().upper())

        # Names that differ only in capitalization are still duplicates.
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(
                "Model names must be unique after converting to uppercase."
            )

        # The IDs are stable with respect to the order supplied by the caller.
        return {
            name: model_id
            for model_id, name in enumerate(normalized_names)
        }

    @staticmethod
    def _build_shared_modules(
        model_config: EINNModelConfig,
        model_names: Sequence[str],
        device: str,
        seed: int = 42,
    ) -> tuple[nn.ModuleDict, FeatureModule, OutputModule, dict[str, int]]:
        """
        Build TimeModules for each model and shared Feature and Output modules.

        Each model gets its own TimeModule. The FeatureModule and OutputModule
        are shared across the requested models, which lets the multi-model
        experiment use one common feature space and one common state decoder.

        :param EINNModelConfig model_config: Neural architecture configuration.
        :param Sequence[str] model_names: Ordered model identifiers.
        :param str device: Target PyTorch device, such as cpu or cuda:0.
        :param int seed: Base seed for TimeModule Fourier feature initialization.
        :return tuple[nn.ModuleDict, FeatureModule, OutputModule, dict[str, int]]:
            Model-specific TimeModules, shared FeatureModule, shared OutputModule,
            and the model-to-label mapping.
        """
        # Normalize once so the module names and label IDs use the same keys.
        names = [name.strip().upper() for name in model_names]
        model_to_id = EINNBuilder.validate_model_registry(names)

        # The feature module's label embedding needs one ID per model.
        if model_config.num_models != len(names):
            raise ValueError(
                f"model_config.num_models={model_config.num_models}, "
                f"but {len(names)} models were requested."
            )

        # Build one time encoder per model. Offset each seed so the Fourier
        # feature bases start independently rather than identically.
        time_modules = nn.ModuleDict(
            {
                name: TimeModule(
                    mapping_size=model_config.time_mapping_size,
                    scale=model_config.time_scale,
                    out_dim=model_config.d_e,
                    seed=seed + model_id,
                )
                for model_id, name in enumerate(names)
            }
        ).to(device)

        # This one feature encoder is shared by the models in a pair. Its label
        # embedding lets it condition representations on the selected model.
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

        # The shared output module maps either model's embedding into the
        # common compartment layout configured by d_s.
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
        Assemble a pair-compatible multi-model EINN.

        Supported pairs are SIR/SIS and SIR/SEIR. Each pair uses a unified
        state layout so the pair can share one OutputModule:

        - SIR/SIS uses d_s=3 and d_p=2.
        - SIR/SEIR uses d_s=4 and d_p=3.

        For SIR/SIS, this method uses SIRSISCompatibleModel from
        einn_multimodel.ode. It does not use the standalone SISModel from
        einn.ode.

        :param EINNModelConfig model_config: Neural and shared ODE dimensions.
        :param EINNTrainConfig train_config: Training configuration.
        :param Mapping[str, Mapping[str, Any]] param_calibration: Initial ODE
            parameters keyed by model name.
        :param float population_n: Population size used by the ODE models.
        :param int seed: Base seed for TimeModule initialization.
        :param Sequence[str] model_names: The two model names in label order.
        :return EINNModels: Initialized multi-model container.
        """
        # Use one canonical spelling for all subsequent lookups.
        names = [name.strip().upper() for name in model_names]

        # Validate the names before constructing modules or ODE objects.
        # This also rejects duplicate names such as ("SIR", "sir").
        EINNBuilder.validate_model_registry(names)

        if len(names) != 2:
            raise ValueError("Exactly two model names are required.")

        # Constructing the pair values locally avoids module-level configuration
        # constants and makes the supported pairs explicit at the use site.
        sir_sis_pair = frozenset(("SIR", "SIS"))
        sir_seir_pair = frozenset(("SIR", "SEIR"))
        selected_pair = frozenset(names)

        if selected_pair not in {sir_sis_pair, sir_seir_pair}:
            raise ValueError(
                "Supported model pairs are SIR/SIS and SIR/SEIR."
            )

        # Both members of a pair must produce states with the same shape.
        if selected_pair == sir_sis_pair:
            expected_d_s = 3
            expected_d_p = 2
        else:
            expected_d_s = 4
            expected_d_p = 3

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

        # The synthetic data generator used by this experiment creates four
        # covariates and one noisy infection channel.
        if model_config.d_x != 5:
            raise ValueError(
                "SyntheticDataGenerator produces five input features "
                "(four covariates plus noisy I); set d_x=5."
            )

        if model_config.num_models != 2:
            raise ValueError(
                "A two-model experiment requires num_models=2."
            )

        # Require exactly one parameter dictionary for each selected model.
        # This catches both missing calibration and misspelled model keys.
        normalized_calibration = {
            name.strip().upper(): values
            for name, values in param_calibration.items()
        }
        if set(normalized_calibration) != set(names):
            raise ValueError(
                "param_calibration must contain exactly "
                f"{sorted(names)}."
            )

        # Build the shared neural network components only after the input
        # configuration has passed validation.
        (
            time_modules,
            feature_module,
            output_module,
            model_to_id,
        ) = EINNBuilder._build_shared_modules(
            model_config=model_config,
            model_names=names,
            device=train_config.device,
            seed=seed,
        )

        # Select the ODE implementation for each supported pair. In the
        # SIR/SIS case, the compatible SIS class is required because its state
        # vector must match the three-state SIR output.
        if selected_pair == sir_sis_pair:
            ode_factories = {
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
            # The compatible SIR model inserts the dummy exposed state so its
            # output layout matches the four-state SEIR model.
            ode_factories = {
                "SIR": lambda: SIRSEIRCompatibleModel(
                    population_n=population_n,
                    target_idx=2,
                ),
                "SEIR": lambda: SEIRModel(
                    population_n=population_n,
                    target_idx=2,
                ),
            }

        # ModuleDict registers each ODE model as a child module. That ensures
        # its parameters are moved with the model and appear in state_dict().
        ode_models = nn.ModuleDict()

        for name in names:
            ode_model = ode_factories[name]()

            # Each model gets only its own calibration values.
            ode_model.init_params(
                param_dict=dict(normalized_calibration[name])
            )

            # Move learnable ODE parameters to the same device as the neural
            # network modules before returning the assembled container.
            ode_models[name] = ode_model.to(train_config.device)

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
        Build a single-model EINN using the standalone ODE implementations.

        This method remains for backwards compatibility with code that uses
        EINNBuilder to build one model. It is separate from build_multi_einn:
        for example, single-model SIS uses einn.ode.SISModel, while the
        multi-model SIR/SIS pair uses SIRSISCompatibleModel.

        :param EINNModelConfig model_config: Neural network and ODE dimensions.
        :param EINNTrainConfig train_config: Training configuration.
        :param Mapping[str, Any] param_calibration: Initial ODE parameters.
        :param str model_type: ODE model name: SIR, SIS, SEIR, or SEIRM.
        :param float | None population_n: Total population size.
        :param int seed: Seed for TimeModule Fourier feature initialization.
        :param int | None target_idx: Target state index. If omitted, the
            default index for the selected model is used.
        :return EINNModels: A single-model container with a one-entry registry.
        :raises ValueError: If the model name, state dimension, or target index
            is invalid.
        """
        # Normalize the identifier before looking it up in the local factory
        # table. The table is local so this file has no module-level registry.
        normalized_model_type = model_type.strip().upper()

        # A missing population size means use normalized population fractions.
        population = 1.0 if population_n is None else float(population_n)

        # Default targets correspond to the infected compartment in each
        # standalone model's state ordering.
        default_target_indices = {
            "SIR": 1,
            "SIS": 1,
            "SEIR": 2,
            "SEIRM": 2,
        }

        # These factories deliberately reference the single-model ODE classes.
        # The pair-compatible classes belong only to build_multi_einn.
        ode_factories = {
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

        if normalized_model_type not in ode_factories:
            supported_types = ", ".join(ode_factories)
            raise ValueError(
                f"Unsupported ODE model type {model_type!r}. "
                f"Supported types are: {supported_types}."
            )

        # Let callers override the target, but provide a sensible model-specific
        # default for existing calls that do not pass target_idx.
        if target_idx is None:
            selected_target_idx = default_target_indices[
                normalized_model_type
            ]
        else:
            selected_target_idx = target_idx

        # Instantiate the selected standalone ODE class.
        ode_model = ode_factories[normalized_model_type](
            selected_target_idx
        )

        # The OutputModule and ODE model must agree on the number of states.
        if model_config.d_s != ode_model.d_s:
            raise ValueError(
                f"{normalized_model_type} has d_s={ode_model.d_s}, "
                f"but model_config.d_s={model_config.d_s}."
            )

        # Reject out-of-range target indices before training or inference.
        if not 0 <= selected_target_idx < ode_model.d_s:
            raise ValueError(
                f"target_idx={selected_target_idx} is outside the valid "
                f"range for {normalized_model_type} with "
                f"d_s={ode_model.d_s}."
            )

        # Initialize the model's raw learnable parameters from physical values.
        ode_model.init_params(param_dict=dict(param_calibration))
        ode_model = ode_model.to(train_config.device)

        # A single model still uses a ModuleDict so it matches the container
        # interface used by the multi-model training and evaluation code.
        time_module = TimeModule(
            mapping_size=model_config.time_mapping_size,
            scale=model_config.time_scale,
            out_dim=model_config.d_e,
            seed=seed,
        ).to(train_config.device)

        # The FeatureModule has one model label in this single-model route.
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

        # Decode the latent representation into the state dimension expected
        # by the selected standalone ODE model.
        output_module = OutputModule(
            d_e=model_config.d_e,
            d_s=model_config.d_s,
        ).to(train_config.device)

        # Build the same name-to-ID mapping used in the multi-model route.
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
