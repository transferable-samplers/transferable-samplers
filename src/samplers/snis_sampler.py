from typing import Optional

import torch

from src.samplers.base_sampler import BaseSampler
from src.samplers.filtering import filter_by_logw_quantile
from src.samplers.resampling import resampling_idx
from src.utils.dataclasses import SamplesData, SourceEnergy, TargetEnergy
from src.utils.dist_utils import all_gather_cat, broadcast_tensor, get_rank, get_world_size
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class SNISSampler(BaseSampler):
    """Self-Normalized Importance Sampling.

    Generates proposal samples, computes importance weights, and resamples.
    Optionally applies logit clipping.
    """

    def __init__(
        self,
        num_samples: int,
        logw_quantile_filter: Optional[float] = None,
    ):
        super().__init__(num_samples)
        self.logw_quantile_filter = logw_quantile_filter

    @torch.no_grad()  # sampling path, no training gradients needed
    def sample(
        self,
        source_energy: SourceEnergy,
        target_energy: TargetEnergy,
    ) -> dict[str, SamplesData]:

        # Generate proposal
        world_size = get_world_size()
        loc_num_samples = self.num_samples // world_size
        loc_samples, loc_E_source = source_energy.sample(loc_num_samples)

        # Compute energy (on each rank, for local samples)
        loc_E_target = target_energy.energy(loc_samples)

        # All gather across ranks
        samples = all_gather_cat(loc_samples)
        E_source = all_gather_cat(loc_E_source)
        E_target = all_gather_cat(loc_E_target)

        # Store for evaluation / plotting
        proposal_data = SamplesData(samples, E_target)

        # Clip by logit quantile (all ranks do same filtering to maintain same shape)
        if self.logw_quantile_filter is not None:
            samples, E_source, E_target = filter_by_logw_quantile(samples, E_source, E_target, self.logw_quantile_filter)
            logger.info("Clipped proposal logw for SMC initialisation")

        # Compute importance weights on all ranks
        logw = E_source - E_target

        # Only resample on rank 0, then broadcast to all ranks
        if get_rank() == 0:
            resampling_index = resampling_idx(logw, "multinomial")
        else:
            resampling_index = torch.zeros(len(logw), dtype=torch.long, device=logw.device)
        resampling_index = broadcast_tensor(resampling_index, src=0)

        resampled_data = SamplesData(
            samples[resampling_index],
            E_target[resampling_index],
            logw=logw,
        )

        return {"proposal": proposal_data, "resampled": resampled_data}, None
