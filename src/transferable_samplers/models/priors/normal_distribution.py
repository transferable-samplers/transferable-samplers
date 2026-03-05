from __future__ import annotations

import torch
from torch.distributions import Normal

from transferable_samplers.models.priors.prior import Prior


class NormalDistribution(Prior):
    """Isotropic Gaussian prior with optional zero center-of-mass constraint.

    When ``mean_free=True``, sampled conformations are projected to have zero
    center of mass, and log-probabilities are computed after the same projection.

    Args:
        num_dimensions: Spatial dimensions per atom (typically 3).
        mean: Mean of the Gaussian.
        std: Standard deviation of the Gaussian.
        mean_free: If True, subtract center of mass from samples and before
            computing log-probabilities.
    """

    def __init__(self, num_dimensions: int = 3, mean: float = 0.0, std: float = 1.0, mean_free: bool = False) -> None:
        self.num_dimensions = num_dimensions
        self.mean = mean
        self.std = std
        self.mean_free = mean_free

        self.distribution = Normal(mean, std)

    def sample(
        self, num_samples: int, num_atoms: int, mask: torch.Tensor | None = None, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        x = self.distribution.sample((num_samples, num_atoms, self.num_dimensions)).to(device)
        if self.mean_free:
            if mask is None:
                mask = torch.ones((num_samples, num_atoms), device=device)
            # pyrefly: ignore [no-matching-overload]
            com = (x * mask[..., None]).sum(dim=1, keepdims=True) / mask.sum(dim=1, keepdims=True)[..., None]
            x = x - com
            x *= mask[..., None]
        return x.reshape(num_samples, num_atoms, self.num_dimensions)

    def logp(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.dim() == 3
        num_samples = x.shape[0]
        num_atoms = x.shape[1]
        if mask is None:
            mask = torch.ones((num_samples, num_atoms), device=x.device)
        if self.mean_free:
            # pyrefly: ignore [no-matching-overload]
            com = (x * mask[..., None]).sum(dim=1, keepdims=True) / mask.sum(dim=1, keepdims=True)[..., None]
            x = x - com
            x *= mask[..., None]

        pointwise_logp = self.distribution.log_prob(x)

        pointwise_logp = pointwise_logp * mask.unsqueeze(-1)
        pointwise_logp = pointwise_logp.reshape(num_samples, -1)
        logp = pointwise_logp.sum(dim=-1)

        return logp
