from typing import Optional

import torch


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, transform=None, inject_metadata: Optional[dict] = None):
        assert isinstance(data, torch.Tensor), f"data must be a torch.Tensor, got {type(data)}"
        self.data = data
        self.transform = transform
        self.inject_metadata = inject_metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {"x": self.data[idx].float()}
        if self.inject_metadata is not None:
            sample.update(self.inject_metadata)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
