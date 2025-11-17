import torch

# TODO this should be factored together with peptides_dataset.py


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].float()
        if self.transform is not None:
            sample = self.transform(
                {
                    "x": x,
                },
            )
        return sample
