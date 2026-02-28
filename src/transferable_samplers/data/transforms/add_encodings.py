from typing import Any


class AddEncodingsTransform:
    """Adds encodings to the data dictionary based on the sequence name."""

    def __init__(self, encodings_dict: dict[str, Any]) -> None:
        self.encodings_dict = encodings_dict

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "mask" not in data, "data should be unpadded (so without a mask)"
        assert data["x"].ndim == 2, f"AddEncodingsTransform only handles single samples, got shape {data['x'].shape}"

        sequence = data["sequence"]
        return {
            **data,
            "encodings": self.encodings_dict[sequence],
        }
