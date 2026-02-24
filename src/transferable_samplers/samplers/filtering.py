from typing import Tuple

import torch


def filter_by_energy_cutoff(
    samples: torch.Tensor, E_source: torch.Tensor, E_target: torch.Tensor, cutoff: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop samples whose target energy exceeds cutoff."""
    keep = E_target < cutoff
    return samples[keep], E_source[keep], E_target[keep]


def filter_by_logw_quantile(
    samples: torch.Tensor, E_source: torch.Tensor, E_target: torch.Tensor, clip_fraction: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop the top clip_fraction of samples by importance weight logit."""
    logw = E_source - E_target
    keep = logw <= torch.quantile(logw, 1 - clip_fraction)
    return samples[keep], E_source[keep], E_target[keep]
