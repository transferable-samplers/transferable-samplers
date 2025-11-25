from abc import ABC, abstractmethod
import torch

from typing import Optional, Callable

import time


from src.utils.dataclasses import SamplesData, SystemConditioning, ProposalModel
from src.utils.batching_utils import tree_cat

class Sampler(ABC):
    def __init__(self, num_samples: int, batch_size: int, use_com_energy_adjustment: bool = False, logit_clip_filter_pct: float = 0.0, log_image_fn: callable = None):
        # Batch size is essentially how many samples the machine can handle in parallel for FWD or REV
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.logit_clip_filter_pct = logit_clip_filter_pct # Applied to the initial proposal samples
        self.log_image_fn = log_image_fn

    def get_resampling_index(logits):
        probs = torch.softmax(logits, dim=-1)
        resampling_index = torch.multinomial(probs, logits.size(0), replacement=True)
        return resampling_index

    @abstractmethod
    def sample(self, proposal_model, target_energy_fn, system_cond: Optional[dict[str, torch.Tensor]] = None, all_gather_fn: Callable[[torch.Tensor], torch.Tensor] = None, prefix: str = "") -> SamplesData:
        raise NotImplementedError
