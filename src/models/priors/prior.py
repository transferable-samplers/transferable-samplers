import torch
from torch.distributions import Normal

from abc import ABC, abstractmethod
class Prior(ABC):
    @abstractmethod
    def sample(self, num_samples: int, num_atoms: int, mask: torch.Tensor | None = None, device="cpu") -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def energy(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
