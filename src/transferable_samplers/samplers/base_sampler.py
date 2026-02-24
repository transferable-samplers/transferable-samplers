from abc import ABC, abstractmethod
from typing import Any

from transferable_samplers.utils.dataclasses import SamplesData, SourceEnergy, TargetEnergy


class BaseSampler(ABC):
    """Base class for sampling strategies.

    Samplers are decoupled from the LightningModule. They receive:
    - source_energy: SourceEnergy with sample() and energy() callables.
    - target_energy: TargetEnergy with energy() callable.
    """

    def __init__(self, num_samples: int):
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
