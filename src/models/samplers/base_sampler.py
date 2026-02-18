from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from tqdm import tqdm

from src.utils.dataclasses import ProposalCond, SamplesData

if TYPE_CHECKING:
    from lightning import LightningModule


class BaseSampler(ABC):
    """Base class for sampling strategies.

    Samplers encapsulate the full sampling pipeline including DDP coordination.
    The model is passed to sample() for access to sample_proposal(), all_gather(), etc.
    """

    def __init__(self, num_samples: int, proposal_batch_size: int):
        self.num_samples = num_samples
        self.proposal_batch_size = proposal_batch_size

    @abstractmethod
    def sample(
        self,
        model: "LightningModule",
        proposal_cond: Optional[ProposalCond],
        target_energy_fn,
    ) -> dict[str, SamplesData]:
        """Run the full sampling pipeline.

        Args:
            model: LightningModule with sample_proposal() and proposal_energy() methods.
            proposal_cond: Optional conditioning (permutations, encodings).
            target_energy_fn: Maps normalized samples to energy scalars.

        Returns:
            Dict mapping sample set names to SamplesData, e.g.
            {"proposal": ..., "resampled": ..., "smc": ...}.
        """
        ...

    @torch.no_grad()
    def sample_proposal_in_batches(
        self,
        model: "LightningModule",
        num_samples: int,
        proposal_cond: Optional[ProposalCond],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate num_samples total across all DDP ranks, in batches.

        Calls model.sample_proposal() in batches on the local rank,
        then all_gathers results across DDP ranks.

        Returns:
            (samples, log_q) gathered across all ranks.
        """
        world_size = model.trainer.world_size if model.trainer else 1
        local_total = num_samples // world_size

        all_samples, all_log_q = [], []
        remaining = local_total
        pbar = tqdm(total=local_total, desc="Generating proposals")
        while remaining > 0:
            batch_n = min(self.proposal_batch_size, remaining)
            samples, log_q = model.sample_proposal(batch_n, proposal_cond)
            all_samples.append(samples)
            all_log_q.append(log_q)
            remaining -= batch_n
            pbar.update(batch_n)
        pbar.close()

        local_samples = torch.cat(all_samples, dim=0)
        local_log_q = torch.cat(all_log_q, dim=0)

        # Gather across DDP ranks
        if world_size > 1:
            local_samples = model.all_gather(local_samples).reshape(-1, *local_samples.shape[1:])
            local_log_q = model.all_gather(local_log_q).reshape(-1, *local_log_q.shape[1:])

        return local_samples, local_log_q
