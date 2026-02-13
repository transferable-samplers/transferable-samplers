import math

import scipy.special
import torch


def resample_multinomial(x, logw):
    """Multinomial resampling. Returns (resampled_x, indexes)."""
    w = torch.softmax(logw, dim=-1)
    indexes = torch.multinomial(w, len(x), replacement=True)
    return x[indexes], indexes


def resample_systematic(x, logw):
    """Systematic resampling. Returns (resampled_x, indexes)."""
    N = len(logw)
    w = torch.softmax(logw, dim=-1)
    c = torch.cumsum(w, dim=0)
    u = torch.rand(1, device=x.device) / N
    indexes = torch.searchsorted(c, u + torch.arange(N, device=x.device) / N)
    return x[indexes], indexes


def com_energy_adjustment(x: torch.Tensor, com_norm_std: float) -> torch.Tensor:
    """Compute center-of-mass energy adjustment.
    x: (batch, num_atoms, 3), com_norm_std: scalar std of CoM norms.
    Introduced in Prop. 1 of https://arxiv.org/pdf/2502.18462
    """
    com = x.mean(dim=1)
    com_norm = com.norm(dim=-1)
    return com_norm**2 / (2 * com_norm_std**2) - torch.log(
        com_norm**2 / (math.sqrt(2) * com_norm_std**3 * scipy.special.gamma(3 / 2))
    )
