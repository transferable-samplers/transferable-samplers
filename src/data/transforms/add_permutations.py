from typing import Any


class AddPermutationsTransform:
    """Adds permutations to the data dictionary based on the sequence name."""

    def __init__(self, permutations_dict: dict[str, Any]) -> None:
        self.permutations_dict = permutations_dict

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "mask" not in data, "data should be unpadded (so without a mask)"
        assert data["x"].ndim == 2, f"AddPermutationsTransform only handles single samples, got shape {data['x'].shape}"

        sequence = data["sequence"]
        return {
            **data,
            "permutations": self.permutations_dict[sequence],
        }
