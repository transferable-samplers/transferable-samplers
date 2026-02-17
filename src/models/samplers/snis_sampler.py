import logging
from typing import TYPE_CHECKING, Optional

import torch

from src.data.normalization import unnormalize
from src.models.samplers.base_sampler_class import BaseSampler
from src.utils.dataclasses import ProposalCond, SamplesData
from src.utils.resampling import com_energy_adjustment, resample_multinomial

if TYPE_CHECKING:
    from lightning import LightningModule

logger = logging.getLogger(__name__)


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
        model: "LightningModule",
        proposal_cond: Optional[ProposalCond],
        target_energy_fn,
        prefix: str = "",
    ) -> dict[str, SamplesData]:
        samples, log_q = self.sample_proposal_in_batches(model, self.num_samples, proposal_cond)
        target_energy = target_energy_fn(samples)

        std = model.trainer.datamodule.std
        proposal_data = SamplesData(unnormalize(samples, std), target_energy)

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
            unnormalize(samples[resampling_index], std),
            target_energy[resampling_index],
            logits=logits,
        )

        return {"proposal": proposal_data, "resampled": resampled_data}
