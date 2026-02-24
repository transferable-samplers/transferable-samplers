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
    """Undo standardization: multiply by std.

    Args:
        x: Standardized tensor of shape (num_samples, num_atoms, num_dimensions).
        std: Scalar standard deviation tensor.

    Returns:
        Destandardized tensor of the same shape.
    """
    assert std.numel() == 1, "Standard deviation should be scalar"
    assert len(x.shape) == 3, "Input should be 3D tensor"
    return x * std.to(x)
