import torch


class EarlyStopping:
    """
    Monitors a specified training/validation metric and stops the training process early if the metric stops improving,
     preventing overfitting and saving computational resources.

    The code is from: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    """
    def __init__(self, mode: str = 'min', min_delta: float = 0, patience: int = 10, percentage: bool = False):

        """
        Initializes the EarlyStopping mechanism.

        :param str mode: 'min' if the objective is to minimize the metric (e.g., loss),
                         'max' if maximizing (e.g., accuracy). Default is 'min'.
        :param float min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param int patience: Number of consecutive epochs with no improvement after which training will be stopped.
        :param bool percentage: If True, `min_delta` is treated as a percentage of the best value.
        """
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        # If patience is 0, EarlyStopping is effectively disabled
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics: float | torch.Tensor) -> bool:
        """
        Evaluates the new metric and updates the early stopping state.

        :param float | torch.Tensor metrics: The latest calculated metric (e.g., validation loss).
        :return bool: True if the training should be stopped (patience exceeded or NaN detected), False otherwise.
        """
        if self.best is None:
            self.best = metrics
            return False

        # Failsafe: if the network diverges and produces NaN, stop immediately
        if torch.is_tensor(metrics) and torch.isnan(metrics):
            return True
        elif not torch.is_tensor(metrics) and torch.tensor(metrics).isnan():
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode: str, min_delta: float, percentage: bool):
        """
        Dynamically constructs the comparison function based on the selected mode and delta settings.

        :param str mode: 'min' or 'max', indicating the direction of improvement.
        :param float min_delta: The threshold for measuring the new optimum.
        :param bool percentage: Whether the min_delta is a percentage or an absolute value.
        """
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)
