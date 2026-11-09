"""Standalone smoke test for evaluator and plotter using lightweight mock models."""
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import torch

# Stub project-only imports so the numerical evaluator can be checked without
# the user's full EINN source tree.
mods = {
    "einn": types.ModuleType("einn"),
    "einn.data": types.ModuleType("einn.data"),
    "einn.inference": types.ModuleType("einn.inference"),
    "einn.model": types.ModuleType("einn.model"),
    "einn.model.interface": types.ModuleType("einn.model.interface"),
    "einn.data.data_scaling_stats": types.ModuleType("einn.data.data_scaling_stats"),
    "einn.data.input_loader": types.ModuleType("einn.data.input_loader"),
    "einn.inference.inference_engine": types.ModuleType("einn.inference.inference_engine"),
    "einn.model.interface.einn_models": types.ModuleType("einn.model.interface.einn_models"),
}

@dataclass
class DataScalingStats:
    aux_mean: torch.Tensor
    aux_std: torch.Tensor
    t_min: torch.Tensor
    t_max: torch.Tensor
    initial_states: list[float]

@dataclass
class GeneratedSeries:
    x_raw: np.ndarray
    y_raw: np.ndarray
    aux_raw: np.ndarray
    true_latent_states: np.ndarray
    true_params: dict
    correlations: dict

class FakeInferenceEngine:
    def __init__(self, models):
        self.models = models
    def predict_future(self, historical_x, historical_x_mask, future_t, scaling_stats):
        batch, length, _ = future_t.shape
        return torch.zeros(batch, length, 5, device=future_t.device)

mods["einn.data.data_scaling_stats"].DataScalingStats = DataScalingStats
mods["einn.data.input_loader"].GeneratedSeries = GeneratedSeries
mods["einn.inference.inference_engine"].InferenceEngine = FakeInferenceEngine
mods["einn.model.interface.einn_models"].EINNModels = object
mods["einn"].__path__ = [str(Path(__file__).parent / "einn")]
mods["einn.data"].__path__ = []
mods["einn.inference"].__path__ = []
mods["einn.model"].__path__ = []
mods["einn.model.interface"].__path__ = []
sys.modules.update(mods)

from einn.evaluation.generalization_evaluator import EINNGeneralizationEvaluator
from einn.visualization.experiment_plotter import EINNExperimentPlotter

class EmbeddingModule:
    def eval(self):
        return self
    def __call__(self, **kwargs):
        if "t" in kwargs:
            t = kwargs["t"]
            return torch.cat([t] * 20, dim=-1)
        x = kwargs["x"]
        t = kwargs["t"]
        base = x.mean(dim=(1, 2), keepdim=True)
        return torch.sin(t * (base + 1.0)) * torch.ones(1, 1, 20, device=t.device)

class FakeModel:
    def eval(self):
        return self

class FakeModels:
    time_module = EmbeddingModule()
    feature_module = EmbeddingModule()
    output_module = FakeModel()
    ode_model = FakeModel()

class FakeScaler:
    def transform(self, x):
        return x

seq_len = 200
time = np.arange(seq_len)
infected = 0.01 + 0.35 * np.exp(-((time - 65) / 28) ** 2)
true_states = np.column_stack([
    0.99 - 0.5 * infected,
    0.0 * infected,
    infected,
    0.5 * infected,
    0.0 * infected,
]).astype(np.float32)
x_raw = np.random.default_rng(7).normal(size=(seq_len, 5)).astype(np.float32)
series = GeneratedSeries(x_raw, infected[:, None].astype(np.float32), np.zeros((seq_len, 5), np.float32), true_states, {}, {})
stats = DataScalingStats(
    aux_mean=torch.zeros(5),
    aux_std=torch.ones(5),
    t_min=torch.tensor(0.0),
    t_max=torch.tensor(float(seq_len - 1)),
    initial_states=[0.99, 0.01, 0.0, 0.0, 0.0],
)

evaluator = EINNGeneralizationEvaluator(device="cpu", target_idx=2)
evaluation = evaluator.evaluate(FakeModels(), series, FakeScaler(), stats)
assert len(evaluation.phases) == 4
assert evaluation.phases[0].feature_embeddings.shape == (seq_len, 20)

out = Path("/tmp/einn_analysis_smoke")
plotter = EINNExperimentPlotter(out)
plotter.plot_generalization_grid(evaluation, d_e=20)
plotter.plot_true_vs_predicted(evaluation.true_states, evaluation.full_prediction)
plotter.save_evaluation(evaluation)
assert (out / "generalization_metrics.json").exists()
assert (out / "generalization_latents.npz").exists()
print("analysis smoke test: OK")
