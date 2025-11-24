from statistics import median

import numpy as np
import torch


def resample(samples, logits, return_index=False):
    """
    Resample samples with given logits.
    Args:
        samples: Samples to resample
        logits: Logits for resampling
    Returns:
        Resampled samples
    """
    probs = torch.softmax(logits, dim=-1)
    resampled_samples = torch.multinomial(probs, samples.size(0), replacement=True)
    return samples[resampled_samples], resampled_samples

def logit_clip_filter(logits, clip_threshold):
    clipped_logits_mask = resampling_logits > torch.quantile(
        resampling_logits,
        1 - float(self.hparams.sampling_config.clip_reweighting_logits),
    )
    proposal_samples = proposal_samples[~clipped_logits_mask]
    proposal_samples_energy = proposal_samples_energy[~clipped_logits_mask]
    resampling_logits = resampling_logits[~clipped_logits_mask]
    logging.info("Clipped logits for resampling")

class SNISSampler(ProposalSampler):
    def __init__(self, use_com_adjustment: bool = False, clip_reweighting_logits: float = 0.0)
        super().__init__(num_samples, output_dir, local_rank)
        self.use_com_adjustment = use_com_adjustment
        self.clip_reweighting_logits = clip_reweighting_logits

    def sample(self, proposal_generator, source_energy, target_energy):

        proposal_data, metrics = super().sample(proposal_generator, source_energy, target_energy)

        # Compute proposal center of mass std
        coms = proposal_data.samples.mean(dim=1, keepdim=False)
        proposal_com_std = coms.std()

        if self.use_com_adjustment:
            proposal_log_q = proposal_data.log_q_theta + com_energy_adjustment(proposal_data.samples, proposal_com_std)

        proposal_samples_energy = target_energy(proposal_data.samples)

        # Compute resampling index
        # proposal_log_p - proposal_log_q
        resampling_logits = -proposal_samples_energy - proposal_log_q

        # Filter samples based on logit clipping - this affects both IS and SMC
        if self.clip_reweighting_logits:
            proposal_data.samples, proposal_data.energy, resampling_logits = logit_clip_filter(proposal_data.samples, proposal_data.energy, resampling_logits)

        _, resampling_index = resample(proposal_data.samples, resampling_logits, return_index=True)

        reweighted_data = SamplesData(
            self.datamodule.unnormalize(proposal_data.samples[resampling_index]),
            proposal_data.energy[resampling_index],
            logits=resampling_logits,
        )


