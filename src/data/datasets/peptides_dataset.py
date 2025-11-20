from typing import Union

import numpy as np
import torch


class PeptideDataset(torch.utils.data.Dataset):
    """
    Unified dataset class for handling both plain tensors and dict-formatted data.
    
    - If data is a plain tensor, each item is wrapped in a dict with "x" key
    - If data is a list of dicts, items are used as-is
    """
    def __init__(
        self,
        data: Union[torch.Tensor, list],
        transform=None,
        sequence: str = None,
    ):
        self.data = data
        self.transform = transform
        self.sequence = sequence
        
        # Determine if data is plain tensors or dict-formatted
        self.is_dict_data = isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_dict_data:
            # Data is already in dict format (peptides case)
            sample = self.data[idx]
            assert "sequence" in sample, "Dict data must contain 'sequence' key"
        else:
            # Data is plain tensors (single peptide case)
            x = self.data[idx].float()
            sample = {"x": x}
            if self.sequence is not None:
                sample["sequence"] = self.sequence

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class PeptideDatasetWithBuffer(torch.utils.data.Dataset):
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


def build_peptide_dataset(
    file_paths: list[str],
    num_dimensions: int = 3,
    num_aa_min: int = None,
    num_aa_max: int = None,
    transform=None,
):
    """
    Build a PeptideDataset from a list of NPZ file paths.

    Args:
        file_paths: List of paths to NPZ files containing peptide data.
        num_dimensions: Number of spatial dimensions (default: 3).
        num_aa_min: Minimum amino acid sequence length to include (optional).
        num_aa_max: Maximum amino acid sequence length to include (optional).
        transform: Optional transform to apply to each sample.

    Returns:
        PeptideDataset: Dataset containing all samples from the provided files.
    """
    assert file_paths, "file_paths list cannot be empty"

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

    return PeptideDataset(data=data, transform=transform)
