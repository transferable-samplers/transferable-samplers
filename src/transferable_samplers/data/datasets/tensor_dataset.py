from __future__ import annotations

from typing import Any

import torch


class TensorDataset(torch.utils.data.Dataset):
    """Map-style dataset backed by a single tensor.

    Args:
        data: Tensor of conformations, shape ``(N, num_atoms, 3)``.
        transform: Optional transform applied to each sample dict.
        inject_metadata: Optional static metadata merged into every sample.
    """

    def __init__(
        self, data: torch.Tensor, transform: Any | None = None, inject_metadata: dict[str, Any] | None = None
    ) -> None:
        assert isinstance(data, torch.Tensor), f"data must be a torch.Tensor, got {type(data)}"
        self.data = data
        self.transform = transform
        self.inject_metadata = inject_metadata

    def __len__(self) -> int:
        return len(self.data)

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = {"x": self.data[idx].float()}
        if self.inject_metadata is not None:
            sample.update(self.inject_metadata)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
