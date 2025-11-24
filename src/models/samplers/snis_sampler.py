from statistics import median

import numpy as np
import torch

class SNISSampler(BaseSampler):
    def sample(self, proposal_samples, proposal_energy_fn, target_energy_fn):
        """
        NOTE: In theory the samplers could share values for x_proposal_energy and x_target_energy,
        But I forsee potential bugs so keep them independent.
        """

        x = proposal_samples

        x_proposal_energy = proposal_energy_fn(x)
        x_target_energy = target_energy_fn(x)

        samples_data = SamplesData(x, x_proposal_energy, x_target_energy)

        # Compute resampling index
        resampling_logits = x_proposal_energy - x_target_energy

        if self.logit_clip_filter_pct > 0:
            logit_clip_mask = self.get_logit_clip_mask(resampling_logits, self.logit_clip_filter_pct)
            samples_data = samples_data[logit_clip_mask] # Filter the samples data based on the logit clipping

        resampling_index = get_resampling_index(resampling_logits)
        samples_data = samples_data[resampling_index] # Resample the samples data based on the resampling index

        return samples_data
        