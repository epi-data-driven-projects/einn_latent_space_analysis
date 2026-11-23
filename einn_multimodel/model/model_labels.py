"""Shared model-label registry for multi-model EINN experiments."""

MODEL_TO_ID: dict[str, int] = {
    "SIR": 0,
    "SIS": 1,
    "SEIR": 2,
}
ID_TO_MODEL: dict[int, str] = {value: key for key, value in MODEL_TO_ID.items()}


def validate_model_registry(model_names: list[str]) -> dict[str, int]:
    """Return contiguous experiment-local labels for the selected models."""
    if not model_names:
        raise ValueError("At least one ODE model must be supplied.")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Model names must be unique.")

    unknown = set(model_names).difference(MODEL_TO_ID)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")

    # Labels must be contiguous within an experiment. For example, an SIR/SEIR
    # experiment uses labels {SIR: 0, SEIR: 1}, not {SIR: 0, SEIR: 2}.
    return {name: idx for idx, name in enumerate(model_names)}