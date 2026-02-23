from typing import Optional

import torch

from src.models.samplers.base_sampler import BaseSampler
from src.models.samplers.utils import filter_by_logit_quantile, resampling_idx
from src.utils import pylogger
from src.utils.dataclasses import SamplesData, SourceEnergy, TargetEnergy
from src.utils.dist_utils import all_gather_cat, broadcast_tensor, get_rank

logger = pylogger.RankedLogger(__name__, rank_zero_only=False)


class SNISSampler(BaseSampler):
    """Self-Normalized Importance Sampling.

    Generates proposal samples, computes importance weights, and resamples.
    Optionally applies logit clipping.
    """

    def __init__(
        self,
        num_samples: int,
        logit_quantile_filter: Optional[float] = None,
    ):
        super().__init__(num_samples)
        self.logit_quantile_filter = logit_quantile_filter

    @torch.no_grad()
    def sample(
        self,
        source_energy: SourceEnergy,
        target_energy: TargetEnergy,
    ) -> dict[str, SamplesData]:

        # Generate proposal
        samples, log_q = source_energy.sample(self.num_samples)

        # Compute energy (on each rank, for local samples)
        target_E = target_energy.energy(samples)

        # All gather samples, log_q, target_E across ranks
        samples = all_gather_cat(samples)
        log_q = all_gather_cat(log_q)
        target_E = all_gather_cat(target_E)

        # Store for evaluation / plotting
        proposal_data = SamplesData(samples, target_E)

        # Importance weights (logits already global — derived from all-gathered samples/log_q)
        logits = -target_E - log_q

        # Clip logits
        if self.logit_quantile_filter is not None:
            samples, log_q, target_E = filter_by_logit_quantile(samples, log_q, target_E, self.logit_quantile_filter)
            # Recompute logits after filtering
            logits = -target_E - log_q
            logger.info("Clipped logits for resampling")

        # Only resample on rank 0, then broadcast to all ranks
        if get_rank() == 0:
            resampling_index = resampling_idx(logits, "multinomial")
        else:
            resampling_index = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        resampling_index = broadcast_tensor(resampling_index, src=0)

        resampled_data = SamplesData(
            samples[resampling_index],
            target_E[resampling_index],
            logits=logits,
        )

        return {"proposal": proposal_data, "resampled": resampled_data}, None
