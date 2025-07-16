from typing import Any

import torch


class StandardizeTransform(torch.nn.Module):
    """Zero center of mass and normalize the coordinates of the molecule."""

    def __init__(self, std: float) -> None:
        """
        Args:
            std (float): Standard deviation for normalization.
        """
        super().__init__()
        self.std = std

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data: The input data dictionary containing (at least) the key "x".
        Returns:
            data: The updated data dictionary with standardized coordinates."
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        x = data["x"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"

        # Calculate the current center of mass
        center_of_mass = x.mean(dim=0)
        x = (x - center_of_mass) / self.std

        return {
            **data,
            "x": x,
        }
