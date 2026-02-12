from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

# See: https://github.com/Lightning-AI/pytorch-lightning/pull/20545
# Important notes:
# The WeightAveraging callback will store in a ckpt:
# - the averaged model weights under ["state_dict"]
# - the non-averaged model weights under ["current_model_state"]
# When resuming training, it is important to resume using the same WeightAveraging callback,
# otherwise the averaged model weights will be loaded as the current parameter state.
# The averaged model weights will be used during validation and testing.


class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay: float = 0.999):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 100 steps.
        return (step_idx is not None) and (step_idx >= 100)
