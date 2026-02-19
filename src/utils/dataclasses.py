from dataclasses import dataclass
from typing import Callable, Optional

import torch

@dataclass
class DistOps:
    """Distributed operations and metadata needed by samplers.

    Created via BaseLightningModule.build_dist_ops(). Decouples samplers
    from the LightningModule so they only depend on these operations.
    """
    world_size: int
    local_rank: int
    all_gather: Callable  # (tensor) -> gathered tensor


@dataclass
class ProposalModel:
    """Wraps proposal sampling and energy computation with system_cond and net pre-bound.

    Created via BaseLightningModule.build_proposal_model(). Samplers receive this
    instead of the raw model + system_cond, allowing EMA weights to be used without
    mutating the model's state.
    """
    sample_proposal: Callable  # (num_samples, log_metrics=True) -> (samples, log_q)
    proposal_energy: Callable  # (x) -> energy

@dataclass
class SystemCond:
    permutations: Optional[dict] = None
    encodings: Optional[dict] = None


@dataclass
class EvalContext:
    true_samples: torch.Tensor
    target_energy_fn: Callable
    system_cond: Optional[SystemCond]
    tica_model: Optional[object] = None
    topology: Optional[object] = None

@dataclass
class SamplesData:
    samples: torch.Tensor
    energy: torch.Tensor
    logits: torch.Tensor = None

    def __post_init__(self):
        assert len(self.samples) == len(self.energy)
        if self.logits is not None:
            assert len(self.samples) == len(self.logits)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return SamplesData(
            self.samples[index],
            self.energy[index],
            self.logits[index] if self.logits is not None else None,
        )


