import json
import logging
import os
import random
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from einn.config.einn_model_config import EINNModelConfig
from einn.config.einn_train_config import EINNTrainConfig
from einn.data.data_preprocessor import DataPreprocessor
from einn.data.data_scaling_stats import DataScalingStats
from einn.data.input_loader import GeneratedSeries, InputLoader
from einn.data.einn_window_generator import EINNWindowGenerator
from einn.data.windowed_einn_dataset import WindowedEINNDataset
from einn.inference.inference_engine import InferenceEngine
from einn.model.einn_builder import EINNBuilder
from einn.model.interface.einn_models import EINNModels
from einn.training.einn_trainer import EINNTrainer
from einn.ode.base_ode_model import BaseODEModel


@dataclass
class EINNRunResult:
    models: EINNModels
    metrics: list
    preprocessor: DataPreprocessor
    scaling_stats: DataScalingStats
    test_series: GeneratedSeries
    predicted_states: np.ndarray
    output_dir: str


class EINNRunner:
    """Coordinate synthetic-data generation, preprocessing, training and inference.

    This runner uses the project's existing model builder, four-phase trainer,
    ODE solution generator (through InputLoader), and InferenceEngine. It adds
    only the experiment-level orchestration and the expanding-prefix dataset.
    """

    DEFAULT_ODE_INITIAL_PARAMETERS = {
        "SIR": {"beta": 0.4, "gamma": 0.1},
        "SEIRM": {"beta": 0.4, "alpha": 0.2, "gamma": 0.1, "mu": 0.05},
    }

    def __init__(
        self,
        output_dir: str | Path,
        true_model: BaseODEModel,
        assumed_model: BaseODEModel,
        model_config: EINNModelConfig,
        train_config: EINNTrainConfig,
        n_train_series: int = 20,
        seq_len: int = 200,
        min_sequence_length: int = 20,
        forecast_horizon: int = 56,
        stride: int = 1,
        true_initial_states: Optional[list[float]] = None,
        assumed_initial_states: Optional[list[float]] = None,
        true_parameter_ranges: Optional[dict[str, tuple[float, float]]] = None,
        parameter_names: Optional[list[str]] = None,
        correlations: Optional[dict[str, float]] = None,
        correlation_ranges: Optional[dict[str, tuple[float, float]]] = None,
        noise_std: float = 0.02,
        calibration_epochs: int = 100,
        calibration_lr: float = 0.01,
        calibration_initial_guess: Optional[list[float]] = None,
        ode_initial_parameters: Optional[dict[str, float]] = None,
        seed: int = 42,
        cache_filename: str = "generated_training_data.pt",
        calibration_device: str = "cpu",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._configure_logger(self.output_dir)

        self.true_model = true_model
        self.assumed_model = assumed_model
        self.model_config = model_config
        self.train_config = train_config
        self.n_train_series = n_train_series
        self.seq_len = seq_len
        self.min_sequence_length = min_sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.seed = seed
        self.assumed_initial_states = list(
            assumed_initial_states or InputLoader._default_initial_states(assumed_model.d_s)
        )

        self.input_loader = InputLoader(
            true_model=true_model,
            assumed_model=assumed_model,
            n_series=n_train_series,
            seq_len=seq_len,
            true_initial_states=true_initial_states,
            assumed_initial_states=self.assumed_initial_states,
            parameter_ranges=true_parameter_ranges,
            parameter_names=parameter_names,
            correlations=correlations,
            correlation_ranges=correlation_ranges,
            noise_std=noise_std,
            calibration_epochs=calibration_epochs,
            calibration_lr=calibration_lr,
            calibration_initial_guess=calibration_initial_guess,
            cache_path=self.output_dir / "data_cache" / cache_filename,
            seed=seed,
            calibration_device=calibration_device,
        )

        model_type = self._get_builder_model_type(assumed_model)
        self.ode_initial_parameters = (
            ode_initial_parameters
            or self.DEFAULT_ODE_INITIAL_PARAMETERS[model_type].copy()
        )

        if min_sequence_length > seq_len:
            raise ValueError("min_sequence_length cannot exceed seq_len.")
        if forecast_horizon < 0:
            raise ValueError("forecast_horizon cannot be negative.")

        # The engine evaluates time values beyond the final observed point.
        self.train_config.future_steps = forecast_horizon
        self.train_config.device = self._resolve_device(self.train_config.device)

    @staticmethod
    def _configure_logger(output_dir: Path) -> logging.Logger:
        logger = logging.getLogger("EINNRunner")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        file_handler = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Use the requested device if available, otherwise fall back safely."""
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                logging.getLogger("EINNRunner").warning(
                    "Requested device %s is unavailable; falling back to CPU.", requested
                )
                return "cpu"
            if ":" in requested:
                index = int(requested.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    logging.getLogger("EINNRunner").warning(
                        "Requested device %s does not exist; using cuda:0.", requested
                    )
                    return "cuda:0"
        return requested

    @staticmethod
    def _get_builder_model_type(model: BaseODEModel) -> str:
        name = model.__class__.__name__.lower()
        if "seirm" in name:
            return "SEIRM"
        if "sir" in name:
            return "SIR"
        raise ValueError(
            f"EINNBuilder currently supports SIR and SEIRM, not {model.__class__.__name__}."
        )

    @staticmethod
    def seed_everything(seed: int) -> None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_scaled_series(
        self,
        records: list[GeneratedSeries],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], DataPreprocessor, DataScalingStats]:
        x_concat = np.concatenate([item.x_raw for item in records], axis=0)
        y_concat = np.concatenate([item.y_raw for item in records], axis=0)
        aux_concat = np.concatenate([item.aux_raw for item in records], axis=0)

        preprocessor = DataPreprocessor()
        _, scaling_stats = preprocessor.fit_transform(
            x_raw=x_concat,
            y_raw=y_concat,
            aux_raw=aux_concat,
            target_idx=self.assumed_model.target_idx,
            initial_states=self.assumed_initial_states,
        )

        # Each trajectory has its own local time axis 0..seq_len-1. The
        # preprocessor saw concatenated rows only to fit pooled feature/state
        # scalers, so replace its concatenated-row time bounds with local bounds.
        preprocessor.t_min = torch.tensor(0.0, dtype=torch.float32)
        preprocessor.t_max = torch.tensor(float(self.seq_len - 1), dtype=torch.float32)
        scaling_stats.t_min = preprocessor.t_min
        scaling_stats.t_max = preprocessor.t_max

        target_idx = self.assumed_model.target_idx
        target_mean = preprocessor.aux_mean[target_idx]
        target_std = preprocessor.aux_std[target_idx]
        # The OutputModule predicts all compartments in aux_scaler's coordinate
        # system. Therefore y must use the same coordinate for the target channel.
        preprocessor.y_mean = target_mean.reshape(1).clone()
        preprocessor.y_std = target_std.reshape(1).clone()

        x_scaled_series: list[np.ndarray] = []
        y_scaled_series: list[np.ndarray] = []
        aux_scaled_series: list[np.ndarray] = []

        for record in records:
            x_scaled = preprocessor.x_scaler.transform(record.x_raw).astype(np.float32)

            aux_for_scaling = record.aux_raw.copy()
            aux_for_scaling[:, target_idx] = record.y_raw.reshape(-1)
            aux_scaled = preprocessor.aux_scaler.transform(aux_for_scaling).astype(np.float32)
            y_scaled = aux_scaled[:, target_idx : target_idx + 1].copy()

            x_scaled_series.append(x_scaled)
            y_scaled_series.append(y_scaled)
            aux_scaled_series.append(aux_scaled)

        return (
            x_scaled_series,
            y_scaled_series,
            aux_scaled_series,
            preprocessor,
            scaling_stats,
        )

    def _build_models(self, d_x: int) -> EINNModels:
        model_type = self._get_builder_model_type(self.assumed_model)
        assumed_param_names = InputLoader._infer_parameter_names(self.assumed_model)
        config = replace(
            self.model_config,
            d_x=d_x,
            d_s=self.assumed_model.d_s,
            d_p=len(assumed_param_names),
        )
        models = EINNBuilder.build_einn(
            model_config=config,
            train_config=self.train_config,
            param_calibration=self.ode_initial_parameters,
            model_type=model_type,
            seed=self.seed,
        )
        # Builder defaults target_idx to the final compartment for SEIRM.
        # The experiment may instead model the infected compartment.
        models.ode_model.target_idx = self.assumed_model.target_idx
        return models

    def _save_training_artifacts(
        self,
        models: EINNModels,
        metrics: list,
        preprocessor: DataPreprocessor,
        scaling_stats: DataScalingStats,
    ) -> None:
        model_dir = self.output_dir / "models"
        model_dir.mkdir(exist_ok=True)
        for name in ("time_module", "feature_module", "output_module", "ode_model"):
            module = getattr(models, name)
            torch.save(module.state_dict(), model_dir / f"{name}.pth")

        metric_rows = [asdict(item) if is_dataclass(item) else item for item in metrics]
        with open(self.output_dir / "training_metrics.json", "w", encoding="utf-8") as file:
            json.dump(metric_rows, file, indent=2)

        joblib.dump(preprocessor, self.output_dir / "preprocessor.joblib")
        torch.save(
            {
                "aux_mean": scaling_stats.aux_mean.cpu(),
                "aux_std": scaling_stats.aux_std.cpu(),
                "t_min": scaling_stats.t_min.cpu(),
                "t_max": scaling_stats.t_max.cpu(),
                "initial_states": scaling_stats.initial_states,
            },
            self.output_dir / "scaling_stats.pt",
        )

        run_config = {
            "n_train_series": self.n_train_series,
            "seq_len": self.seq_len,
            "min_sequence_length": self.min_sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "stride": self.stride,
            "seed": self.seed,
            "device": self.train_config.device,
            "model_type": self._get_builder_model_type(self.assumed_model),
            "target_idx": self.assumed_model.target_idx,
            "ode_initial_parameters": self.ode_initial_parameters,
            "train_config": asdict(self.train_config),
            "model_config": asdict(self.model_config),
        }
        with open(self.output_dir / "run_config.json", "w", encoding="utf-8") as file:
            json.dump(run_config, file, indent=2, default=str)

    def run_experiment(
        self,
        epochs: int = 1,
        batch_size: int = 32,
        force_regenerate: bool = False,
        test_params: Optional[dict[str, float]] = None,
        test_correlations: Optional[dict[str, float]] = None,
    ) -> EINNRunResult:
        """Run generation/cache -> scaling -> expanding windows -> train -> inference."""
        self.seed_everything(self.seed)
        self.logger.info("1/6: Loading or generating synthetic trajectories and aux data")
        records = self.input_loader.load_or_generate(force_regenerate=force_regenerate)

        self.logger.info("2/6: Fitting pooled scalers and transforming trajectories")
        (
            x_scaled,
            y_scaled,
            aux_scaled,
            preprocessor,
            scaling_stats,
        ) = self._prepare_scaled_series(records)

        self.logger.info("3/6: Creating original-style expanding-prefix windows")
        window_generator = EINNWindowGenerator(
            min_sequence_length=self.min_sequence_length,
            forecast_horizon=self.forecast_horizon,
            stride=self.stride,
        )
        dataset = WindowedEINNDataset(
            x_series=x_scaled,
            y_series=y_scaled,
            aux_series=aux_scaled,
            window_generator=window_generator,
        )
        self.logger.info("Created %d training windows", len(dataset))

        self.logger.info("4/6: Building EINN")
        models = self._build_models(d_x=x_scaled[0].shape[1])

        self.logger.info("5/6: Training EINN")
        trainer = EINNTrainer(
            models=models,
            config=self.train_config,
            scaling_stats=scaling_stats,
        )
        metrics = trainer.train(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
        )

        self.logger.info("6/6: Saving models and running held-out inference")
        test_series = self.input_loader.generate_test_series(
            true_params=test_params,
            correlations=test_correlations,
        )
        x_test_scaled = preprocessor.x_scaler.transform(test_series.x_raw).astype(np.float32)
        historical_x = torch.as_tensor(
            x_test_scaled,
            dtype=torch.float32,
            device=self.train_config.device,
        ).unsqueeze(0)
        historical_mask = torch.ones(
            historical_x.shape[:2],
            dtype=torch.float32,
            device=self.train_config.device,
        )
        future_t = torch.arange(
            self.seq_len,
            dtype=torch.float32,
            device=self.train_config.device,
        ).view(1, self.seq_len, 1)

        inference_engine = InferenceEngine(models)
        predicted = inference_engine.predict_future(
            historical_x=historical_x,
            historical_x_mask=historical_mask,
            future_t=future_t,
            scaling_stats=scaling_stats,
        )
        predicted_states = predicted.squeeze(0).detach().cpu().numpy()

        self._save_training_artifacts(
            models=models,
            metrics=metrics,
            preprocessor=preprocessor,
            scaling_stats=scaling_stats,
        )
        np.savez_compressed(
            self.output_dir / "held_out_prediction.npz",
            x_raw=test_series.x_raw,
            y_raw=test_series.y_raw,
            true_latent_states=test_series.true_latent_states,
            predicted_states=predicted_states,
            true_params=test_series.true_params,
            correlations=test_series.correlations,
        )

        self.logger.info("Experiment finished. Results saved to %s", self.output_dir)
        return EINNRunResult(
            models=models,
            metrics=metrics,
            preprocessor=preprocessor,
            scaling_stats=scaling_stats,
            test_series=test_series,
            predicted_states=predicted_states,
            output_dir=str(self.output_dir),
        )
