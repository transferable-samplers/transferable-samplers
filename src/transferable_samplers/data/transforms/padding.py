from typing import Any

import torch


class PaddingTransform:
    """Pads the input data to a fixed size and creates a mask for the padded elements."""

    def __init__(self, max_num_atoms: int) -> None:
        self.max_num_atoms = max_num_atoms

    def pad_data(self, x: torch.Tensor) -> torch.Tensor:
        """Pad a 2D tensor to max_num_atoms along the first dimension."""
        assert len(x.shape) == 2
        num_atoms = x.shape[0]
        assert num_atoms <= self.max_num_atoms, f"number of particles {num_atoms} exceeds max {self.max_num_atoms}"
        pad_tensor = torch.zeros(self.max_num_atoms - num_atoms, x.shape[1], dtype=x.dtype)
        return torch.cat([x, pad_tensor])

    def pad_encodings(self, encodings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pad encoding tensors to max_num_atoms, leaving seq_len unchanged."""
        padded_encodings = {}
        for key, value in encodings.items():
            if not key == "seq_len":  # don't pad seq_len - is single value per sample
                padded_encodings[key] = torch.cat(
                    [value, torch.zeros(self.max_num_atoms - value.shape[0], dtype=torch.int64)]
                )
            else:
                padded_encodings[key] = value
        return padded_encodings

    def pad_permutations(self, permutations: dict[str, torch.Tensor], padded_seq_len: int) -> dict[str, torch.Tensor]:
        """Pad permutation tensors with sequential indices up to padded_seq_len."""
        num_tokens = next(iter(permutations.values())).shape[0]
        assert all(len(v) == num_tokens for v in permutations.values()), "All permutations must have same length"
        pad_len = padded_seq_len - num_tokens
        assert pad_len >= 0
        if not pad_len:
            return permutations.copy()
        else:
            padded_permutations = {}
            for key, value in permutations.items():
                pad_start = torch.max(value).item() + 1
                pad_values = torch.arange(pad_start, pad_start + pad_len, dtype=torch.int64)
                padded_permutations[key] = torch.cat([value, pad_values])
            return padded_permutations

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pad data, encodings, and permutations, and add a boolean mask."""
        assert "mask" not in data, "data already has a mask, cannot pad again"

        x = data["x"]
        encodings = data["encodings"]
        permutations = data.get("permutations")

        assert x.ndim == 2, f"PaddingTransform only handles single samples, got shape {x.shape}"

        num_atoms = x.shape[0]
        mask = torch.cat(
            [
                torch.ones(num_atoms, dtype=torch.bool),
                torch.zeros(self.max_num_atoms - num_atoms, dtype=torch.bool),
            ]
        )

        x = self.pad_data(x)
        encodings = self.pad_encodings(encodings)

        padded_batch = {
            **data,
            "x": x,
            "encodings": encodings,
            "mask": mask,
        }

        if permutations is not None:
            padded_batch["permutations"] = self.pad_permutations(permutations, padded_seq_len=self.max_num_atoms)

        return padded_batch
