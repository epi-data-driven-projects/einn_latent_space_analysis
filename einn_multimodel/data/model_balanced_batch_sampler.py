"""
Label-balanced PyTorch batch sampler for multi-model EINN training with SupConLoss.
"""
from collections import defaultdict
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler


class ModelBalancedBatchSampler(Sampler[list[int]]):
    """
    PyTorch BatchSampler balancing every batch across all (true_model, assumed_model) pairs.

    Ensures each batch contains at least min_per_true_class samples per true epidemiological
    model family so the Supervised Contrastive loss (SupConLoss, thesis Section 2.4) always
    has non-empty positive pairs P(i) for every anchor, while equally representing matched
    and mismatched model-assumption cells when present.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        min_per_true_class: int = 2,
    ) -> None:
        """
        Partition dataset indices by (true_model_label, assumed_model_label) and validate batching constraints.

        Each cell contributes per_cell = batch_size // num_cells samples per batch, yielding
        effective batches of length per_cell * num_cells (<= batch_size).

        :param Any dataset: ExpandingWindowLoader-like dataset exposing a non-empty .samples
            sequence where each element provides 'true_model_label' and 'assumed_model_label'.
        :param int batch_size: Target number of samples B per batch.
        :param int min_per_true_class: Minimum number of samples required from each true model
            class per batch (default 2, guaranteeing at least one positive pair per anchor in SupCon).
        :return None: Initializes cell index partitions and per-cell quota in place.
        :raises TypeError: If dataset does not expose a non-empty samples attribute.
        :raises ValueError: If fewer than 2 true model classes exist, batch_size is smaller
            than the number of cells, any cell has fewer than per_cell samples, or any true
            class contributes fewer than min_per_true_class samples per batch.
        """
        if not hasattr(dataset, "samples") or not dataset.samples:
            raise TypeError("Dataset must expose non-empty samples for label-aware batching.")

        # Group dataset indices into cells keyed by (true_model_label, assumed_model_label).
        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        true_classes: set[int] = set()
        for index, sample in enumerate(dataset.samples):
            true_label = int(torch.as_tensor(sample["true_model_label"]).item())
            assumed_label = int(torch.as_tensor(sample["assumed_model_label"]).item())
            cells[(true_label, assumed_label)].append(index)
            true_classes.add(true_label)

        if len(true_classes) < 2:
            raise ValueError("SupCon batching requires at least two true-model classes.")
        if len(cells) > batch_size:
            raise ValueError(
                f"batch_size={batch_size} is too small to include all {len(cells)} "
                "true/assumed combinations."
            )

        self.cells: dict[tuple[int, int], list[int]] = dict(cells)
        self.batch_size: int = batch_size
        self.per_cell: int = batch_size // len(cells)
        if any(len(indices) < self.per_cell for indices in self.cells.values()):
            raise ValueError(
                "Every true/assumed label combination must contain at least "
                f"{self.per_cell} samples."
            )

        # Verify each true model class meets the minimum positive-pair count across its cells.
        per_true: dict[int, int] = defaultdict(int)
        for (true_label, _), indices in self.cells.items():
            per_true[true_label] += min(len(indices), self.per_cell)
        if any(count < min_per_true_class for count in per_true.values()):
            raise ValueError(
                "Each true-model class must contribute at least two examples per batch. "
                "Increase batch_size or provide more data per label combination."
            )

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yield randomly shuffled, cell-balanced lists of dataset indices for one epoch.

        Independently permutes each (true_model, assumed_model) cell, draws self.per_cell
        indices from every cell per batch, and shuffles the combined batch order.

        :return Iterator[list[int]]: Iterator yielding index lists of length self.per_cell * len(self.cells).
        """
        # Independently permute indices within each (true_model, assumed_model) cell.
        shuffled = {
            key: torch.randperm(len(indices)).tolist()
            for key, indices in self.cells.items()
        }
        cursors = {key: 0 for key in self.cells}

        # Number of complete balanced batches is bounded by the smallest cell.
        batches = min(len(indices) // self.per_cell for indices in self.cells.values())
        for _ in range(batches):
            batch: list[int] = []
            for key, indices in self.cells.items():
                start = cursors[key]
                selected = shuffled[key][start:start + self.per_cell]
                cursors[key] += self.per_cell
                batch.extend(indices[position] for position in selected)

            # Shuffle within-batch ordering so model classes are interleaved.
            order = torch.randperm(len(batch)).tolist()
            yield [batch[position] for position in order]

    def __len__(self) -> int:
        """
        Return the number of complete balanced batches yielded per epoch.

        :return int: Minimum floor(len(cell) / self.per_cell) across all label cells.
        """
        return min(len(indices) // self.per_cell for indices in self.cells.values())
