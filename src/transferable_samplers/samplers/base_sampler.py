from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from transferable_samplers.utils.dataclasses import SamplesData, SourceEnergy, TargetEnergy


class BaseSampler(ABC):
    """Abstract base class for sampling strategies.

    Samplers are decoupled from the LightningModule. They receive
    ``SourceEnergy`` and ``TargetEnergy`` callables and return named
    sample sets with optional diagnostics.

    Subclasses must implement:
        - ``sample``: Run the full sampling pipeline.

    Args:
        num_samples: Total number of samples to generate.
    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    @abstractmethod
    def sample(
        self,
        source_energy: SourceEnergy,
        target_energy: TargetEnergy,
    ) -> tuple[dict[str, SamplesData], Any | None]:
        """Run the full sampling pipeline.

        Args:
            source_energy: SourceEnergy with sample() and energy() callables.
            target_energy: TargetEnergy with energy() callable.

        Returns:
            (samples_dict, diagnostics) — samples_dict maps names to SamplesData,
            diagnostics is sampler-specific output (e.g. SMC trajectory) or None.
        """
        ...
