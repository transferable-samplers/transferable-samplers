from __future__ import annotations

import torch


def ess(log_weights: torch.Tensor) -> torch.Tensor:
    """Kish effective sample size; log weights don't have to be normalized."""
    return torch.exp(2 * torch.logsumexp(log_weights, dim=0) - torch.logsumexp(2 * log_weights, dim=0))


def normalized_ess(log_weights: torch.Tensor) -> torch.Tensor:
    """Kish effective sample size / sample size; log weights don't have to be normalized."""
    return ess(log_weights) / len(log_weights)
