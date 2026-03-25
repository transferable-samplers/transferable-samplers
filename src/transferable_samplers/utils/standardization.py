import torch


def standardize_coords(x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardize trajectory samples: subtract center of mass and divide by std.

    Args:
        x: Tensor of shape (num_samples, num_atoms, num_dimensions).
        std: Scalar standard deviation tensor.

    Returns:
        Standardized tensor of the same shape.
    """
    assert std.numel() == 1, "Standard deviation should be scalar"
    assert len(x.shape) == 3, "Input should be 3D tensor"
    x = x - x.mean(dim=1, keepdim=True)
    x = x / std
    return x


def destandardize_coords(x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Rescale standardized coordinates by multiplying by std.

    This reverses only the scaling applied in :func:`standardize_coords` and does
    not restore the subtracted center of mass. To fully invert the transformation,
    the original center of mass must be added back separately.

    Args:
        x: Standardized tensor of shape (num_samples, num_atoms, num_dimensions),
            typically output from :func:`standardize_coords` (possibly after other
            operations).
        std: Scalar standard deviation tensor.

    Returns:
        Tensor of the same shape, with original scale restored but still centered
        (zero-mean per sample).
    """
    assert std.numel() == 1, "Standard deviation should be scalar"
    assert len(x.shape) == 3, "Input should be 3D tensor"
    return x * std.to(x)
