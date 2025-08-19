import glob
import os

import numpy as np
import torch

# TODO this should be factored together with tensor_dataset.py


class PeptidesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        transform=None,
    ):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        assert "sequence" in sample

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class PeptidesDatasetWithBuffer(torch.utils.data.Dataset):
    def __init__(
        self,
        buffer,
        transform=None,
    ):
        self.buffer = buffer
        self.transform = transform

    def __len__(self):
        # fake max number of samples in buffer
        # to estimate number of training steps
        if len(self.buffer) == 0:
            return self.buffer.max_length

        return len(self.buffer)

    def __getitem__(self, idx):
        if len(self.buffer) == 0:
            raise IndexError("Attempting to get an item from a completely empty dataset.")

        sample = self.sample_buffer(idx)
        assert "sequence" in sample
    
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def sample_buffer(self, idx):
        # Sample by default 1 for __getitem__
        # can sample larger batch_sizes by directly
        # calling this.
        # No need for idx since the input idx will
        # not correspond to the internal idx.
        x, seq_name = self.buffer.sample(idx)
        return {"x": x, "sequence": seq_name}

    def add(self, x, seq_name):
        self.buffer.add(x, seq_name)


def build_peptides_dataset(
    path: str,
    suffix: str = "npz",
    num_dimensions: int = 3,
    num_aa_min: int = None,
    num_aa_max: int = None,
    transform=None,
):
    if os.path.isdir(path):
        file_paths = glob.glob(os.path.join(path, "*/*.npz"))
        assert file_paths, f"No .{suffix} files found!"
    elif os.path.isfile(path):
        file_paths = [path]
    else:
        raise ValueError(f"File or directory {path} is not found!")

    data = []
    for file_path in file_paths:
        file_data = np.load(file_path)
        sequence = str(file_data["sequence"])
        if (num_aa_max is not None) and (len(sequence) > num_aa_max):
            continue
        if (num_aa_min is not None) and (len(sequence) < num_aa_min):
            continue

        positions = torch.from_numpy(file_data["positions"])
        data_dict_list = [{"sequence": sequence, "x": x.reshape(-1, num_dimensions)} for x in positions]
        data.extend(data_dict_list)

    return PeptidesDataset(data=data, transform=transform)
