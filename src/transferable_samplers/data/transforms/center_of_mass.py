import math
from typing import Any

import torch


class CenterOfMassTransform:
    """Applies Gaussian noise to the center of mass of the molecule.

    Supports both single samples (num_atoms, 3) and batched samples (batch, num_atoms, 3).
    Batched inputs are assumed to be unpadded and from the same system (same num_atoms).
    """

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        x = data["x"]
        assert x.ndim in (2, 3), f"expected 2D or 3D tensor, got shape {x.shape}"
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)

        num_atoms = x.shape[1]
        std = 1 / math.sqrt(num_atoms)
        noise = torch.randn(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype) * std
        x = x + noise

        if unbatched:
            x = x.squeeze(0)

        return {**data, "x": x}
