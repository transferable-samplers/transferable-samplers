from typing import Tuple

import torch


def filter_by_energy_cutoff(
    samples: torch.Tensor, log_q: torch.Tensor, target_E: torch.Tensor, cutoff: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop samples whose target energy exceeds cutoff."""
    keep = target_E < cutoff
    return samples[keep], log_q[keep], target_E[keep]


def filter_by_logit_quantile(
    samples: torch.Tensor, log_q: torch.Tensor, target_E: torch.Tensor, clip_fraction: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop the top clip_fraction of samples by importance weight logit."""
    logits = -target_E - log_q
    keep = logits <= torch.quantile(logits, 1 - clip_fraction)
    return samples[keep], log_q[keep], target_E[keep]


def resampling_idx(logw: torch.Tensor, method: str) -> torch.Tensor:
    """Dispatch to the appropriate resampling index function."""
    if method == "multinomial":
        return resampling_idx_multinomial(logw)
    elif method == "systematic":
        return resampling_idx_systematic(logw)
    else:
        raise ValueError(f"Unknown resampling method: {method!r}")


def resampling_idx_multinomial(logw: torch.Tensor) -> torch.Tensor:
    """Multinomial resampling. Returns resampling indices."""
    w = torch.softmax(logw, dim=-1)
    return torch.multinomial(w, len(logw), replacement=True)


def resampling_idx_systematic(logw: torch.Tensor) -> torch.Tensor:
    """Systematic resampling. Returns resampling indices."""
    N = len(logw)
    w = torch.softmax(logw, dim=-1)
    c = torch.cumsum(w, dim=0)
    u = torch.rand(1, device=logw.device) / N
    return torch.searchsorted(c, u + torch.arange(N, device=logw.device) / N)

