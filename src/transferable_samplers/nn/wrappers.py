from __future__ import annotations

from typing import Any

import torch


class TorchDynWrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format.

    When compute_dlogp=True, augments the state with a divergence dimension
    and computes the exact log-density change via Jacobian trace.
    """

    def __init__(self, model: torch.nn.Module, compute_dlogp: bool = True, dlogp_tol_scale: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.nfe = 0
        self.compute_dlogp = compute_dlogp
        self.dlogp_tol_scale = dlogp_tol_scale

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self.compute_dlogp:
            self.nfe += 1
            return self.model(t, x)

        x = x[..., :-1]  # remove the divergence estimate

        dx = self.model(t, x)
        dlogp = -torch.vmap(self._divergence, in_dims=(None, 0), randomness="different")(t.to(x.device).reshape(1), x)

        self.nfe += 1
        return torch.cat([dx, dlogp[:, None] / self.dlogp_tol_scale], dim=-1).detach()

    def _divergence(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        def vecfield(y: torch.Tensor) -> torch.Tensor:
            y = y.view(1, -1)  # batch dims required by EGNN architecture
            return self.model(t, y).flatten()

        jac = torch.func.jacrev(vecfield)
        # pyrefly: ignore [bad-argument-type]
        return torch.trace(jac(x))
