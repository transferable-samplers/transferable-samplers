from typing import Any

import torch


class AddEncodingsTransform(torch.nn.Module):
    """Adds encodings to the data dictionary based on the sequence name."""

    def __init__(self, encodings_dict: dict[str, Any]) -> None:
        """
        Args:
            encodings_dict (Dict[str, Any]): A dictionary mapping sequence names to their respective encodings.
        """
        super().__init__()
        self.encodings_dict = encodings_dict

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): The input data dictionary containing (at least) the key "sequence".
        Returns:
            Dict[str, Any]: The updated data dictionary with the encodings added.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        sequence = data["sequence"]
        return {
            **data,
            "encodings": self.encodings_dict[sequence],
        }
