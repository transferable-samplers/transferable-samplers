from typing import Any


class StandardizeTransform:
    """Zero center of mass and normalize the coordinates of the molecule.

    Supports both single samples (num_atoms, 3) and batched samples (batch, num_atoms, 3).
    Batched inputs are assumed to be unpadded and from the same system (same num_atoms).
    """

    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        x = data["x"]
        assert x.ndim in (2, 3), f"expected 2D or 3D tensor, got shape {x.shape}"
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)

        center_of_mass = x.mean(dim=1, keepdim=True)
        x = (x - center_of_mass) / self.std

        if unbatched:
            x = x.squeeze(0)

        return {**data, "x": x}
