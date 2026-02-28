from __future__ import annotations

import torch


class DummyDataset(torch.utils.data.Dataset):
    """Minimal dataset used as a placeholder for val/test dataloaders.

    All evaluation is handled by custom callbacks (SamplingEvaluationCallback),
    so these dataloaders only exist for Lightning compatibility. Also used as
    a placeholder for buffer-based training — the model owns the buffer and
    overrides batches in training_step().
    """

    # Default size is large enough to be safely divisible across multiple devices.
    def __init__(self, size: int = 256) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": torch.zeros(1)}
