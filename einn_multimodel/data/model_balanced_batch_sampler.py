from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler


class ModelBalancedBatchSampler(Sampler[list[int]]):
    """Build batches balanced over (true model, assumed model) combinations.

    This gives SupCon positive examples for each true class while ensuring that
    matched and mismatched assumptions are represented in the same batch.
    """

    def __init__(self, dataset, batch_size: int, min_per_true_class: int = 2):
        """Initialize ModelBalancedBatchSampler to sample batches balanced by mechanistic model labels.


        :param Any dataset: The prepared padded expanding-window dataset.
        :param int batch_size: Number of samples evaluated in one optimization batch.
        :param int min_per_true_class: Minimum number of examples from each true mechanistic model per batch.
        :return None: No return value; state or serialized artifacts are updated in place.
        """
        if not hasattr(dataset, "samples") or not dataset.samples:
            raise TypeError("Dataset must expose non-empty samples for label-aware batching.")
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

        self.cells = dict(cells)
        self.batch_size = batch_size
        self.per_cell = batch_size // len(cells)
        if any(len(indices) < self.per_cell for indices in self.cells.values()):
            raise ValueError(
                "Every true/assumed label combination must contain at least "
                f"{self.per_cell} samples."
            )

        # At least two examples per true class are required. Since every class
        # has one or more assumed-model cells, this condition is checked below.
        per_true: dict[int, int] = defaultdict(int)
        for (true_label, _), indices in self.cells.items():
            per_true[true_label] += min(len(indices), self.per_cell)
        if any(count < min_per_true_class for count in per_true.values()):
            raise ValueError(
                "Each true-model class must contribute at least two examples per batch. "
                "Increase batch_size or provide more data per label combination."
            )

    def __iter__(self) -> Iterator[list[int]]:
        """Perform iter as part of sample batches balanced by mechanistic model labels.


        :return Iterator[list[int]]: Computed result in the configured output representation.
        """
        shuffled = {
            key: torch.randperm(len(indices)).tolist()
            for key, indices in self.cells.items()
        }
        cursors = {key: 0 for key in self.cells}
        batches = min(len(indices) // self.per_cell for indices in self.cells.values())
        for _ in range(batches):
            batch: list[int] = []
            for key, indices in self.cells.items():
                start = cursors[key]
                selected = shuffled[key][start:start + self.per_cell]
                cursors[key] += self.per_cell
                batch.extend(indices[position] for position in selected)
            order = torch.randperm(len(batch)).tolist()
            yield [batch[position] for position in order]

    def __len__(self) -> int:
        """Perform len as part of sample batches balanced by mechanistic model labels.


        :return int: Computed result in the configured output representation.
        """
        return min(len(indices) // self.per_cell for indices in self.cells.values())
