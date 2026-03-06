from __future__ import annotations

import torch

from transferable_samplers.samplers.base_sampler import BaseSampler
from transferable_samplers.samplers.filtering import filter_by_logw_quantile
from transferable_samplers.samplers.resampling import resampling_idx
from transferable_samplers.utils.dataclasses import SamplesData, SourceEnergy, TargetEnergy
from transferable_samplers.utils.dist_utils import all_gather_cat, broadcast_tensor, get_rank, get_world_size
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class SNISSampler(BaseSampler):
    """Self-Normalized Importance Sampling.

    Generates proposal samples, computes importance weights, and resamples.
    DDP-aware: samples are generated per-rank and gathered globally.

    Args:
        num_samples: Total number of proposal samples to generate.
        logw_quantile_filter: Drop the top fraction of proposals by importance weight.
    """

    def __init__(
        self,
        num_samples: int,
        logw_quantile_filter: float | None = None,
    ) -> None:
        super().__init__(num_samples)
        self.logw_quantile_filter = logw_quantile_filter

    @torch.no_grad()  # sampling path, no training gradients needed
    # pyrefly: ignore [bad-override]
    def sample(
        self,
        source_energy: SourceEnergy,
        target_energy: TargetEnergy,
    ) -> tuple[dict[str, SamplesData], None]:
        """Run SNIS sampling."""
        # Generate proposal
        world_size = get_world_size()
        if self.num_samples % world_size != 0:
            raise ValueError(
                f"SNISSampler.num_samples ({self.num_samples}) must be divisible by "
                f"world_size ({world_size}) to avoid under-sampling."
            )
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
            samples, E_source, E_target = filter_by_logw_quantile(
                samples, E_source, E_target, self.logw_quantile_filter
            )
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

        # pyrefly: ignore [bad-return]
        return {"proposal": proposal_data, "resampled": resampled_data}, None
