import math
from typing import Any

import torch


class CenterOfMassTransform(torch.nn.Module):
    """Applies Gaussian noise to the center of mass of the molecule."""

    def __init__(self) -> None:
        """
        Args:
            num_dimensions (int): Number of dimensions for the atom coordinates. Default is 3.
        """
        super().__init__()

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): The input data dictionary containing (at least) the key "x".
        Returns:
            Dict[str, Any]: The updated data dictionary with added noise to the center of mass.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        x = data["x"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"

        num_atoms = x.shape[0]
        std = 1 / math.sqrt(num_atoms)

        # Generate noise and adjust the center of mass
        noise = torch.randn_like(x[0]) * std

        # Shift all atoms so that the center of mass is moved
        x = x + noise

        return {
            **data,
            "x": x,
        }
