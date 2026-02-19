from typing import Optional

import torch

from src.models.samplers.base_sampler import BaseSampler
from src.utils import pylogger
from src.utils.dataclasses import DistOps, ProposalModel, SamplesData
from src.utils.resampling import com_energy_adjustment, resample_multinomial

logger = pylogger.RankedLogger(__name__, rank_zero_only=False)


class SNISSampler(BaseSampler):
    """Self-Normalized Importance Sampling.

    Generates proposal samples, computes importance weights, and resamples.
    Optionally applies center-of-mass energy adjustment and logit clipping.
    """

    def __init__(
        self,
        num_samples: int,
        proposal_batch_size: int,
        use_com_adjustment: bool = False,
        logit_clip_filter: Optional[float] = None,
    ):
        super().__init__(num_samples, proposal_batch_size)
        self.use_com_adjustment = use_com_adjustment
        self.logit_clip_filter = logit_clip_filter

    @torch.no_grad()
    def sample(
        self,
        proposal_model: ProposalModel,
        target_energy_fn,
        dist_ops: Optional[DistOps] = None,
        log_metrics: bool = True,
    ) -> dict[str, SamplesData]:
        samples, log_q = self.sample_proposal_in_batches(proposal_model, self.num_samples, dist_ops=dist_ops, log_metrics=log_metrics)
        target_energy = target_energy_fn(samples)

        proposal_data = SamplesData(samples, target_energy)

        # CoM adjustment
        if self.use_com_adjustment:
            coms = samples.mean(dim=1)
            com_std = coms.std()
            logger.info(f"Applying CoM energy adjustment (com_std={com_std:.4f})")
            log_q = log_q + com_energy_adjustment(samples, com_std)

        # Importance weights
        logits = -target_energy - log_q

        # Clip logits
        if self.logit_clip_filter:
            clipped_mask = logits > torch.quantile(logits, 1 - self.logit_clip_filter)
            samples = samples[~clipped_mask]
            target_energy = target_energy[~clipped_mask]
            logits = logits[~clipped_mask]
            logger.info("Clipped logits for resampling")

        # Resample
        _, resampling_index = resample_multinomial(samples, logits)

        resampled_data = SamplesData(
            samples[resampling_index],
            target_energy[resampling_index],
            logits=logits,
        )

        return {"proposal": proposal_data, "resampled": resampled_data}
