from typing import Any

import torch
from scipy.spatial.transform import Rotation as R


class Random3DRotationTransform:
    """Applies a random 3D rotation to the input data coordinates.

    Supports both single samples (num_atoms, 3) and batched samples (batch, num_atoms, 3).
    Batched inputs are assumed to be unpadded and from the same system (same num_atoms).
    """

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        x: torch.Tensor = data["x"]
        assert x.ndim in (2, 3), f"expected 2D or 3D tensor, got shape {x.shape}"
        assert x.shape[-1] == 3, f"only 3D rotations are supported, got {x.shape[-1]}"
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        rot = torch.tensor(R.random(batch_size).as_matrix()).to(x)
        x = torch.einsum("bij,bki->bkj", rot, x)

        if unbatched:
            x = x.squeeze(0)

        return {**data, "x": x}
