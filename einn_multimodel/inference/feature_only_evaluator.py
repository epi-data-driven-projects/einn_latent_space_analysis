class FeatureOnlyEvaluator:
    """
    Evaluation orchestrator for multi-model EINN held-out trajectories and latent-space reports.

    Validates consistency between the training-fitted target scaler and auxiliary compartment
    statistics, extracts timestep-level Feature embeddings without temporal pooling (Section 3),
    and delegates figure rendering to MultiModelPlotter.
    """
    def __init__(self):
        pass