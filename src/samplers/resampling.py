import torch


def _resampling_idx_multinomial(logw: torch.Tensor) -> torch.Tensor:
    """Multinomial resampling. Returns resampling indices."""
    w = torch.softmax(logw, dim=-1)
    return torch.multinomial(w, len(logw), replacement=True)


def _resampling_idx_systematic(logw: torch.Tensor) -> torch.Tensor:
    """Systematic resampling. Returns resampling indices."""
    N = len(logw)
    w = torch.softmax(logw, dim=-1)
    c = torch.cumsum(w, dim=0)
    u = torch.rand(1, device=logw.device) / N
    return torch.searchsorted(c, u + torch.arange(N, device=logw.device) / N)


def resampling_idx(logw: torch.Tensor, method: str) -> torch.Tensor:
    """Dispatch to the appropriate resampling index function."""
    if method == "multinomial":
        return _resampling_idx_multinomial(logw)
    elif method == "systematic":
        return _resampling_idx_systematic(logw)
    else:
        raise ValueError(f"Unknown resampling method: {method!r}")
