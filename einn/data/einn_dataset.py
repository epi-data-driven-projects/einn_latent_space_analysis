import torch
from torch.utils.data import Dataset


class EINNDataset(Dataset):
    """
    PyTorch Dataset for EINN. Supports both full-sequence training and a sliding window approach for dynamic batching.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                 aux_targets: torch.Tensor, window_size: int | None = None,
                 device: str = 'cpu'):
        """
        Initializes the dataset, detaches tensors from the computation graph, and moves them to the target device.

        :param torch.Tensor x: the observed inputs. Expected shape: [Seq_len, d_x] or [1, Seq_len, d_x].
        :param torch.Tensor y : the target variable
        :param torch.Tensor t: time
        :param torch.Tensor aux_targets: The ideal trajectories (ODE pre-calibration).
        :param Optional[int] window_size: Size of the sliding window. If None, uses the full sequence.
        :param str device: Device to store the dataset tensors on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)

        # Initializing empty tensors
        self.x = torch.empty(0, device=self.device)
        self.y = torch.empty(0, device=self.device)
        self.t = torch.empty(0, device=self.device)
        self.aux_targets = torch.empty(0, device=self.device)

        # Preparing and loading data
        self._prepare_data(x=x, y=y, t=t, aux_targets=aux_targets)

        # defining sequence length based on the first dimension
        self.seq_len = self.x.size(0)

        # If no window size is provided, default to the entire sequence length (1 large window)
        self.window_size = window_size if window_size is not None else self.seq_len

        # Validation to prevent out-of-bounds slicing errors
        if self.window_size > self.seq_len:
            raise ValueError(f"Window size ({self.window_size}) cannot be " + \
                             f"strictly larger than the sequence length ({self.seq_len}).")

    def _prepare_data(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, aux_targets: torch.Tensor) -> None:
        """

        :param x:
        :param y:
        :param t:
        :param aux_targets:
        :return:
        """
        # Squeeze out the batch dimension if the user passes tensors in [1, Seq_len, Features] format.
        # This makes the slicing logic in __getitem__ cleaner
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
            y = y.squeeze(0)
            t = t.squeeze(0)
            aux_targets = aux_targets.squeeze(0)

        # .detach() ensures no gradients are tracked from previous data processing steps
        # .float() ensures the correct precision for PyTorch default settings.
        # .to(device) moves all data to GPU RAM upfront, maximizing __getitem__ speed during training.
        self.x = x.detach().float().to(self.device)
        self.y = y.detach().float().to(self.device)
        self.t = t.detach().float().to(self.device)
        self.aux_targets = aux_targets.detach().float().to(self.device)

    def __len__(self) -> int:
        """
        Formula: Sequence Length - Window Size + 1
        :return int: the total number of sliding windows available in the sequence.
        """
        return self.seq_len - self.window_size + 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieves the windowed slice of data starting at the given index.

        :param int idx: The starting index of the window.
        :return Dict[str, torch.Tensor]: A dictionary containing the sliced tensors for X, y, t, and aux_targets.
         Shape for each tensor will be [Window_size, Features].
        """
        start_idx = idx
        end_idx = idx + self.window_size

        return {
            'X': self.x[start_idx:end_idx],
            'y': self.y[start_idx:end_idx],
            't': self.t[start_idx:end_idx],
            'aux_targets': self.aux_targets[start_idx:end_idx]
        }
