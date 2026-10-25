from collections import defaultdict

import torch


def collate_all_batches(dataloader):
    collated_batches = defaultdict(list)
    num_batches = 0

    for batch in dataloader:
        num_batches += 1
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Expected tensor values in batch for key '{key}', got {type(value)}"
                )
            collated_batches[key].append(value)

    if num_batches == 0:
        raise ValueError("Cannot collate empty dataloader")

    return {key: torch.cat(values, dim=0) for key, values in collated_batches.items()}
