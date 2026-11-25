> [!IMPORTANT]
> **TDK thesis code — `tdk-dolgozat` branch**
>
> **Magyarul:** A 2026-os TDK-dolgozathoz tartozó forráskód a **`tdk-dolgozat`** branchen található. A dolgozat kísérleteinek reprodukálásához ezt a branchet használd, ne automatikusan az alapértelmezettet.
>
> **English:** The code accompanying the **2026 TDK thesis** is maintained on the **`tdk-dolgozat`** branch. To reproduce the thesis experiments, use that branch rather than assuming the default branch contains the corresponding implementation.
>
> ```bash
> git fetch origin
> git switch tdk-dolgozat
> ```

# EINN — Latent-Space Analysis in Mathematical Epidemiology

**Investigating the latent representations of Epidemiologically-Informed Neural Networks through mechanistic epidemic models, synthetic time series, and geometric analysis.**

This research codebase explores what an **Epidemiologically-Informed Neural Network (EINN)** learns internally, beyond how accurately it predicts an epidemic trajectory. It combines neural time-series modeling with compartmental ordinary differential equations (ODEs), and investigates how the learned embeddings respond to changes in the underlying epidemic model and its parameters.

The work accompanies the 2026 University of Szeged TDK thesis:

> **Látens-tér alapú adatvezérelt modellek beágyazási tereinek vizsgálata a matematikai epidemiológiában**  
> *Kökény Zoltán Tamás — University of Szeged, Bolyai Institute (2026)*

## Contents

- [Research objectives](#research-objectives)
- [Architecture and methodology](#architecture-and-methodology)
- [Experiments](#experiments)
- [Getting started](#getting-started)
- [Running experiments](#running-experiments)
- [Recreating figures from saved results](#recreating-figures-from-saved-results)
- [Quick validation](#quick-validation)
- [Outputs and reproducibility](#outputs-and-reproducibility)
- [Repository structure](#repository-structure)
- [Implementation notes](#implementation-notes)
- [Citation](#citation)

## Research objectives

The primary question is **how the geometry of learned latent representations reflects epidemiological dynamics and model assumptions**.

The code supports investigations of:

- **Representation geometry:** dimensionality, principal-component structure, and the temporal organization of embeddings.
- **Mechanistic consistency:** relationships between the Time and Feature representations and between reconstructed trajectories and their underlying ODEs.
- **Generalization:** how predictions and embeddings change when the observed prefix is shorter than the full simulated trajectory.
- **Model discrimination:** whether embeddings associated with different compartmental models can be distinguished using supervised contrastive learning.

The evaluation includes principal component analysis (PCA), cosine similarity, Procrustes alignment, and—in the multi-model setting—cluster-separation metrics and Hausdorff-type distances.

## Architecture and methodology

The EINN architecture has three principal neural components:

| Component | Role |
|---|---|
| **Time Module** | Learns a time-indexed latent representation, informed by the mechanistic epidemic dynamics. |
| **Feature Module** | Extracts time-dependent embeddings from observed epidemic data and auxiliary covariates. |
| **Output Module** | Maps the embeddings to estimated epidemiological compartment states. |

Both neural branches are trained with data-fitting and ODE-informed objectives. The implementation also includes additional consistency terms, such as initial-condition constraints and, where applicable, monotonicity regularization. The multi-model extension incorporates **supervised contrastive learning (SupCon)** and model-conditioned representations.

### Epidemiological models

The repository implements several compartmental systems, including:

- **SIR** — Susceptible–Infectious–Recovered.
- **SIS** — Susceptible–Infectious–Susceptible.
- **SEIR** — Susceptible–Exposed–Infectious–Recovered; used in the model-comparison experiments.
- **SEIRM** — an additional compartmental model available in the ODE implementation.
- **Röst** — the implemented 18-state epidemiological system inspired by the Röst model, used for higher-dimensional single-model experiments.

## Experiments

The experiment runner supports **11 selectable configurations** across three families.

| Family | Configuration | Latent dimension(s) | Main purpose |
|---|---|---|---|
| **Single-model** | SIR, SIS, Röst | `5`, `20` | Assess representation geometry and generalization under one assumed model. |
| **Dual inference** | SIR: high-β vs. low-β trajectories | `20` | Compare two unseen parameterizations under one trained network. |
| **Multi-model** | SIR–SIS, SIR–SEIR | `5`, `20` | Analyze separation of embeddings from different mechanistic models. |

The single-model and multi-model experiments evaluate four observation-prefix lengths (`10`, `18`, `40`, `80`) on trajectories with 200 time steps. The SIR dual-inference experiment uses a 100-step horizon and two contrasting parameter settings. Evaluations generate diagnostic **4 × 4 figures** and supporting numerical data; the multi-model workflow additionally examines the joint held-out latent space.

### Typical configuration

The TDK-oriented experiment implementations use a fixed random seed (`42`), synthetic ODE data, four training phases, and model-specific noise and parameter settings. The experiment scripts are the authoritative source for the exact configuration:

- `experiments/single_model_experiment.py`
- `experiments/dual_inference_experiment.py`
- `experiments/multi_model_experiment.py`

**Important:** Changing a parameter range, ODE solver, random seed, noise level, or training schedule defines a different experiment. Record such changes when comparing figures or numerical results.

## Getting started

### Requirements

- Python **3.10+** (the implementation uses modern type annotations).
- PyTorch and [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq).
- NumPy, SciPy, scikit-learn, Matplotlib, and Seaborn.
- A CUDA-capable GPU is recommended for training; CPU execution is supported but can be slower.

### Installation

From the repository root, check out the thesis branch and create an isolated environment:

```bash
git fetch origin
git switch tdk-dolgozat

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If your checkout contains `requirements.txt`, install the pinned project dependencies:

```bash
python -m pip install -r requirements.txt
```

Otherwise, install the dependencies listed above, choosing a **PyTorch build compatible with your system/CUDA version**. Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the appropriate command.

All commands below must be run **from the repository root**, so that the local `einn`, `einn_multimodel`, `experiments`, and `entry_points` packages can be imported.

## Running experiments

The main entry point performs **data generation → training → inference → saving → visualization**:

```bash
python -m entry_points.run_experiment \
    --experiment single --model sir --dim 20 --device cuda:1
```

Choose `--device cuda:0`, `cuda:1`, or `cpu` according to your environment.

### Single-model experiments

Run one model at one embedding dimension:

```bash
python -m entry_points.run_experiment \
    --experiment single --model rost --dim 20 --device cuda:1
```

Run **all six** SIR, SIS, and Röst configurations (both embedding dimensions), excluding dual inference:

```bash
python -m entry_points.run_experiment \
    --experiment single --model all --dim all --device cuda:1
```

### Dual inference

Train the specified SIR network and compare its predictions for two epidemic trajectories with different transmission rates:

```bash
python -m entry_points.run_experiment \
    --experiment dual --device cuda:1
```

### Multi-model experiments

Run one model pair:

```bash
python -m entry_points.run_experiment \
    --experiment multi --pair sir-seir --dim 20 --device cuda:1
```

Run all four multi-model configurations:

```bash
python -m entry_points.run_experiment \
    --experiment multi --pair all --dim all --device cuda:1
```

### All experiments

Preview the full run list **without training**:

```bash
python -m entry_points.run_experiment --experiment all --dry-run
```

Execute all **11 configurations** sequentially:

```bash
python -m entry_points.run_experiment \
    --experiment all --device cuda:1
```

## Recreating figures from saved results

A separate entry point regenerates figures using previously saved **numerical arrays**, without retraining the network or running a new inference:

```bash
# Recreate all available single-model reports.
python -m entry_points.plot_saved \
    --experiment single --model all --dim all

# Recreate the dual-inference report.
python -m entry_points.plot_saved --experiment dual

# Recreate the SIR–SEIR reports at d_e = 20.
python -m entry_points.plot_saved \
    --experiment multi --pair sir-seir --dim 20

# Recreate all available reports.
python -m entry_points.plot_saved --experiment all
```

The redraw command expects the numerical files produced by the corresponding experiment runner. **Model checkpoints alone are not sufficient** for saved-data-only plotting. The command writes regenerated figures to separate `redrawn/` directories where supported, helping preserve the original images.

Custom saved-data roots can be passed using `--single-root`, `--dual-root`, and `--multi-root`.

## Quick validation

Before launching long training runs, check that the project imports and the experiment selectors work:

```bash
python -m compileall -q einn einn_multimodel experiments entry_points
python -m unittest discover -s tests -v
python -m entry_points.run_experiment --experiment all --dry-run
```

If `smoke_test_all_einn.py` is present in the repository root, run the lightweight end-to-end test suite:

```bash
python smoke_test_all_einn.py --device cuda:1
```

The smoke test uses reduced training settings while exercising **all 11 configurations**, including data generation, training phases, inference, serialization, and figure creation. It writes per-case logs and a summary under `smoke_test_results/`.

**Validation status:** All 11 configurations have passed a one-epoch smoke test in the project environment. This verifies execution paths and output generation—not convergence, prediction quality, or scientific reproducibility of fully trained models.

## Outputs and reproducibility

By default, results are written to separate directories according to the experiment family:

```text
experiment_results/
├── generalization/
│   ├── sir/de_5/ and sir/de_20/
│   ├── sis/de_5/ and sis/de_20/
│   └── rost/de_5/ and rost/de_20/
└── sir_dual_inference/

outputs/
├── sir_sis_generalization_4x4/
└── sir_seir_generalization_4x4/

smoke_test_results/            # Created by the optional smoke-test script
```

Depending on the experiment, saved artifacts include neural-network checkpoints, reference and reconstructed trajectories, latent embeddings, PCA coordinates and explained variance, numerical metrics, figures, and compressed `.npz` files for reproducible plotting.

To support meaningful comparisons:

1. Use a consistent Git revision from `tdk-dolgozat` and record any local changes.
2. Preserve the experiment's parameter ranges, solver configuration, initial conditions, noise, and seed.
3. Evaluate on held-out trajectories, rather than treating training accuracy as generalization.
4. Compare latent trajectories using a **shared PCA coordinate system** when interpreting their relative geometry.
5. Keep quick smoke-test outputs separate from results obtained after full training.

## Repository structure

```text
.
├── einn/                         # Single-model EINN implementation
│   ├── config/                   # Model and training configuration
│   ├── data/                     # Synthetic data, scaling, windowing
│   ├── model/                    # Time, Feature, and Output modules
│   │   └── interface/           # Typed model-output data structures
│   ├── ode/                      # Compartmental epidemiological ODEs
│   ├── training/                 # Four training phases and optimizers
│   ├── loss/                     # Data, ODE, and regularization losses
│   ├── inference/                # Evaluation and prediction
│   ├── analysis/                 # Loading previously saved arrays
│   ├── visualization/            # Single-model and dual-inference figures
│   └── utils/
├── einn_multimodel/              # Multi-model and SupCon extension
│   ├── config/                   # Multi-model-specific configuration
│   ├── data/                     # Model-labeled synthetic data
│   ├── model/                    # Label-conditioned neural modules
│   ├── ode/                      # Compatible model implementations
│   ├── training/                 # Multi-model optimization
│   ├── loss/                     # Including supervised contrastive loss
│   ├── inference/                # Held-out latent-space evaluation
│   └── visualization/            # Multi-model diagnostic figures
├── experiments/                  # Research experiment implementations
│   ├── single_model_experiment.py
│   ├── dual_inference_experiment.py
│   └── multi_model_experiment.py
├── entry_points/                 # User-facing command-line entry points
│   ├── run_experiment.py
│   └── plot_saved.py                
└── smoke_test_all_einn.py        # Optional fast integration test
```

The modular structure keeps **data preparation, physical modeling, neural-network computation, optimization, evaluation, and plotting** separate. In the refactored implementation, each class is defined in its own Python file, with English docstrings and explicit parameter/return documentation.

## Implementation notes

- **Synthetic-data research:** The experiments use simulated epidemiological trajectories and constructed auxiliary features. The repository is not a calibrated public-health forecasting system.
- **ODE integration:** Broad parameter ranges can produce fast dynamics; an overly large fixed RK4 step may become unstable. A `NaN` trajectory should be investigated at the integration stage, not silently repaired during standardization. Changes to the solver must be documented when comparing experiments.
- **Multi-model assumptions:** Supervised labels and label-conditioned representations are part of the experimental setup; strong classification performance alone does not establish successful mechanistic model identification on real-world data.
- **Comparability:** The code and settings on the `tdk-dolgozat` branch are the relevant reference for the TDK thesis. Other branches may contain later development or experiments with different assumptions.

## Citation

If you use this repository in academic work, please cite the accompanying thesis:

> Kökény, Zoltán Tamás (2026). *Látens-tér alapú adatvezérelt modellek beágyazási tereinek vizsgálata a matematikai epidemiológiában*. TDK thesis, University of Szeged, Bolyai Institute.

```bibtex
@misc{kokeny2026latent,
  author       = {Kökény, Zoltán Tamás},
  title        = {Látens-tér alapú adatvezérelt modellek beágyazási tereinek
                  vizsgálata a matematikai epidemiológiában},
  howpublished = {TDK thesis, University of Szeged, Bolyai Institute},
  year         = {2026}
}
```

---

*Research software developed for methodological investigation of epidemiologically informed neural networks and their latent representation spaces.*