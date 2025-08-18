from typing import Any

import torch


class AddPermutationsTransform(torch.nn.Module):
    """Adds permutations to the data dictionary based on the sequence name."""

    def __init__(self, permutations_dict: dict[str, Any]) -> None:
        """
        Args:
            permutations_dict (Dict[str, Any]): A dictionary mapping sequence names to their respective permutations.
        """
        super().__init__()
        self.permutations_dict = permutations_dict

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): The input data dictionary containing (at least) the key "sequence".
        Returns:
            Dict[str, Any]: The updated data dictionary with the permutations added.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        sequence = data["sequence"]
        return {
            **data,
            "permutations": self.permutations_dict[sequence],
        }
