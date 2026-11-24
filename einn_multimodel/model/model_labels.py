"""
Shared epidemiological model-label registry for multi-model EINN experiments.

Defines the canonical set of supported mechanistic model families (SIR, SIS, SEIR) and
constructs contiguous experiment-local integer ID mappings required by nn.Embedding,
multi-model Time/ODE routing, and SupConLoss (thesis Section 2.4).
"""

# Canonical registry of supported epidemiological models and their global integer IDs.
MODEL_TO_ID: dict[str, int] = {
    "SIR": 0,
    "SIS": 1,
    "SEIR": 2,
}

# Inverse lookup table mapping global integer IDs back to canonical model names.
ID_TO_MODEL: dict[int, str] = {value: key for key, value in MODEL_TO_ID.items()}


def validate_model_registry(model_names: list[str]) -> dict[str, int]:
    """
    Validate requested model names and return a contiguous 0-based experiment-local label map.

    Contiguous indexing (0, 1, ..., K - 1) is required so model labels directly index the
    FeatureModuleEncoder's nn.Embedding(num_models, model_label_dim) table—for example,
    an SIR/SEIR run maps to {'SIR': 0, 'SEIR': 1} rather than {'SIR': 0, 'SEIR': 2}.

    :param list[str] model_names: Non-empty, duplicate-free ordered list of canonical model names
        chosen from MODEL_TO_ID (e.g. ['SIR', 'SIS'] or ['SIR', 'SEIR']).
    :return dict[str, int]: Experiment-local mapping from each model name to its contiguous
        integer ID in [0, len(model_names) - 1].
    :raises ValueError: If model_names is empty, contains duplicates, or includes any name
        not present in MODEL_TO_ID.
    """
    if not model_names:
        raise ValueError("At least one ODE model must be supplied.")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Model names must be unique.")

    unknown = set(model_names).difference(MODEL_TO_ID)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")

    # Assign contiguous 0-based IDs matching the order in model_names.
    return {name: idx for idx, name in enumerate(model_names)}
