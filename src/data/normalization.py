import torch


def normalize(x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize trajectory samples: subtract center of mass and divide by std.

    Args:
        x: Tensor of shape (num_samples, num_atoms, num_dimensions).
        std: Scalar standard deviation tensor.

    Returns:
        Normalized tensor of the same shape.
    """
    assert std.numel() == 1, "Standard deviation should be scalar"
    assert len(x.shape) == 3, "Input should be 3D tensor"
    x = x - x.mean(dim=1, keepdim=True)
    x = x / std
    return x


def unnormalize(x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo normalization: multiply by std.

    Args:
        x: Normalized tensor of shape (num_samples, num_atoms, num_dimensions).
        std: Scalar standard deviation tensor.

    Returns:
        Unnormalized tensor of the same shape.
    """
    assert std.numel() == 1, "Standard deviation should be scalar"
    assert len(x.shape) == 3, "Input should be 3D tensor"
    return x * std.to(x)
