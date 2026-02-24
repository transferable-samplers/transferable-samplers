import torch

from transferable_samplers.utils.dataclasses import SystemCond
from transferable_samplers.utils.standardization import destandardize_coords


class Buffer:
    """Holds resampled training samples with associated system conditioning and transforms.

    Used by BaseLightningModule for self-improvement training. The model populates
    the buffer via its sampler, then draws batches from it during training_step().

    Samples are stored in normalized space and unnormalized internally before
    batch_transform is applied (since transforms expect physical-scale coordinates).
    """

    def __init__(
        self,
        samples: torch.Tensor,
        normalization_std: torch.Tensor,
        system_cond: SystemCond | None,
        batch_transform=None,
    ):
        self.samples = samples
        self.normalization_std = normalization_std
        self.system_cond = system_cond
        self.batch_transform = batch_transform

    def __len__(self):
        return len(self.samples)

    def sample(self, batch_size: int) -> dict:
        """Draw a random batch from the buffer.

        1. Randomly selects samples and stacks into a batch.
        2. Applies batch_transform to the entire batch at once.
        3. Adds system_cond fields (expanded to batch_size).
        """
        indices = torch.randint(0, len(self.samples), (batch_size,))
        batch = {"x": destandardize_coords(self.samples[indices], self.normalization_std)}

        if self.batch_transform is not None:
            batch = self.batch_transform(batch)

        if self.system_cond is not None:
            batched_cond = self.system_cond.for_batch(batch_size)
            if batched_cond.encodings is not None:
                # pyrefly: ignore [unsupported-operation]
                batch["encodings"] = batched_cond.encodings
            if batched_cond.permutations is not None:
                # pyrefly: ignore [unsupported-operation]
                batch["permutations"] = batched_cond.permutations

        return batch
