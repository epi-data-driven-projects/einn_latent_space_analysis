from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    epoch: int
    phase: int
    rep: int

    total_loss: float
