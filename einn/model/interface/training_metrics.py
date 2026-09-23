from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """
    Dataclass for storing and tracking training metrics per epoch, phase, and repetition. Used to log the progression
     of the multi-phase training process and monitor the convergence of the neural network.

    :param int epoch: The training epoch number (starting from 1).
    :param int phase: The specific training phase currently being executed (1, 2, 3, or 4).
    :param int rep: The current repetition (iteration) number within the specified training phase.
    :param float total_loss: The aggregated composite scalar loss, averaged over all
                             processed batches during this specific repetition.
    """
    epoch: int
    phase: int
    rep: int

    total_loss: float
