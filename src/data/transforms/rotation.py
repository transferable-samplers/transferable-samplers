from typing import Any

import torch
from scipy.spatial.transform import Rotation as R


class Random3DRotationTransform(torch.nn.Module):
    """Applies a random 3D rotation to the input data coordinates."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data: The input data dictionary containing (at least) the key "x".
        Returns:
            data: The updated data dictionary with the rotated coordinates.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        x: torch.Tensor = data["x"]
        assert x.shape[1] == 3, f"only 3D rotations are supported, got {x.shape[1]}"

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"

        x = x.unsqueeze(0)
        rot = torch.tensor(R.random(len(x)).as_matrix()).to(x)
        x = torch.einsum("bij,bki->bkj", rot, x)
        x = x.squeeze(0)

        return {
            **data,
            "x": x,
        }
