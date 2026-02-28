from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Prior(ABC):
    @abstractmethod
    def sample(
        self, num_samples: int, num_atoms: int, mask: torch.Tensor | None = None, device: str | torch.device = "cpu"
    ) -> torch.Tensor: ...

    @abstractmethod
    def logp(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor: ...
