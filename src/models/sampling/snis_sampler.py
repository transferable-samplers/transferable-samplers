from typing import Optional

import torch

from src.utils.batching_utils import gather_tree
from src.utils.dataclasses import SamplesData


class SNISSMixin(SamplingMixin):
    def _sample(
        self, num_samples: int, system_cond: Optional[dict[str, torch.Tensor]] = None, prefix: str = ""
    ) -> SamplesData:
        local_num_samples = self.num_samples // world_size
        local_batch_size = self.batch_size // world_size

        metrics = {}

        # TODO rename, or return a dataclass
        proposal_output = self.generate_in_batches(
            proposal_model, local_num_samples, local_batch_size, system_cond=system_cond, all_gather_fn=all_gather_fn
        )
        proposal_x, proposal_log_q_x, proposal_z = gather_tree(proposal_output, all_gather_fn)

        x_proposal_energy = -log_q_x
        if self.use_com_energy_adjustment:
            com_std = self.compute_com_std(x)
            x_proposal_energy += self.com_energy_adjustment_fn(x, com_std)

        x_target_energy = target_energy_fn(x)

        proposal_data = SamplesData(x, x_proposal_energy, x_target_energy)
        temp_data = proposal_data.clone()

        # Compute resampling index
        resampling_logits = x_proposal_energy - x_target_energy

        if self.logit_clip_filter_pct > 0:
            logit_clip_mask = self.get_logit_clip_mask(resampling_logits, self.logit_clip_filter_pct)
            temp_data = temp_data[logit_clip_mask]  # Filter the samples data based on the logit clipping

        resampling_index = get_resampling_index(resampling_logits)
        resampled_data = temp_data[resampling_index]  # Resample the samples data based on the resampling index

        return {
            "proposal": proposal_data,
            "resampled": resampled_data,
        }
