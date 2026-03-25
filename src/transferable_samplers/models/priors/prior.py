from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Prior(ABC):
    """Abstract base class for prior distributions.

    Subclasses must implement:
        - ``sample``: Draw samples from the prior.
        - ``logp``: Compute log-probability of given samples.
    """

    @abstractmethod
    def sample(
        self, num_samples: int, num_atoms: int, mask: torch.Tensor | None = None, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        """Draw samples from the prior distribution.

        Args:
            num_samples: Number of samples to draw.
            num_atoms: Number of atoms per conformation.
            mask: Optional binary mask ``(num_samples, num_atoms)`` for variable-size systems.
            device: Device to place samples on.

        Returns:
            Samples tensor ``(num_samples, num_atoms, num_dimensions)``.
        """
        ...

    @abstractmethod
    def logp(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute log-probability under the prior.

        Args:
            x: Samples tensor ``(num_samples, num_atoms, num_dimensions)``.
            mask: Optional binary mask for variable-size systems.

        Returns:
            Log-probability tensor ``(num_samples,)``.
        """
        ...
