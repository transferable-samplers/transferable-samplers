from abc import ABC, abstractmethod
from typing import Optional

import torch
from tqdm import tqdm

from src.utils.dataclasses import DistOps, ProposalModel, SamplesData


class BaseSampler(ABC):
    """Base class for sampling strategies.

    Samplers are decoupled from the LightningModule. They receive:
    - dist_ops: distributed operations (all_gather, world_size, etc.)
    - proposal_model: pre-bound sample_proposal() and proposal_energy()
    - target_energy_fn: maps normalized samples to energy scalars
    """

    def __init__(self, num_samples: int, proposal_batch_size: int):
        self.num_samples = num_samples
        self.proposal_batch_size = proposal_batch_size

    @abstractmethod
    def sample(
        self,
        proposal_model: ProposalModel,
        target_energy_fn,
        dist_ops: Optional[DistOps] = None,
        log_metrics: bool = True,
    ) -> dict[str, SamplesData]:
        """Run the full sampling pipeline.

        Args:
            dist_ops: Distributed operations and metadata.
            proposal_model: Pre-bound proposal with sample_proposal() and proposal_energy().
            target_energy_fn: Maps normalized samples to energy scalars.
            log_metrics: Whether to log metrics. Must be False when
                called outside valid logging contexts (e.g. on_fit_start).

        Returns:
            Dict mapping sample set names to SamplesData, e.g.
            {"proposal": ..., "resampled": ..., "smc": ...}.
        """
        ...

    @torch.no_grad()
    def sample_proposal_in_batches(
        self,
        proposal_model: ProposalModel,
        num_samples: int,
        dist_ops: Optional[DistOps] = None,
        log_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate num_samples total across all DDP ranks, in batches.

        Calls proposal_model.sample_proposal() in batches on the local rank,
        then all_gathers results across DDP ranks.

        Returns:
            (samples, log_q) gathered across all ranks.
        """
        world_size = dist_ops.world_size if dist_ops else 1
        local_total = num_samples // world_size

        all_samples, all_log_q = [], []
        remaining = local_total
        pbar = tqdm(total=local_total, desc="Generating proposals")
        while remaining > 0:
            batch_n = min(self.proposal_batch_size, remaining)
            samples, log_q = proposal_model.sample_proposal(batch_n, log_metrics=log_metrics)
            all_samples.append(samples)
            all_log_q.append(log_q)
            remaining -= batch_n
            pbar.update(batch_n)
        pbar.close()

        local_samples = torch.cat(all_samples, dim=0)
        local_log_q = torch.cat(all_log_q, dim=0)

        # Gather across DDP ranks
        if dist_ops and dist_ops.world_size > 1:
            local_samples = dist_ops.all_gather(local_samples).reshape(-1, *local_samples.shape[1:])
            local_log_q = dist_ops.all_gather(local_log_q).reshape(-1, *local_log_q.shape[1:])

        return local_samples, local_log_q
