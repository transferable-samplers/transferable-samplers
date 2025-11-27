from abc import ABC, abstractmethod

import torch


class SamplingMixin(ABC):
    @abstractmethod
    def _sample(num_samples: int, system_cond: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
