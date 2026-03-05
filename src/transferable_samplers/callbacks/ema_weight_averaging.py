from __future__ import annotations

from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn


class EMAWeightAveraging(WeightAveraging):
    """Maintain an EMA of model weights, updated every training step.

    Checkpoints store:
        - ``["state_dict"]``: the EMA-averaged weights (used for val/test).
        - ``["current_model_state"]``: the non-averaged training weights.

    When resuming, the same callback must be present so that the averaged and
    current weights are restored correctly.

    See: https://github.com/Lightning-AI/pytorch-lightning/pull/20545

    Args:
        decay: EMA decay factor. Higher means slower averaging.
    """

    def __init__(self, decay: float = 0.999) -> None:
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx: int | None = None, epoch_idx: int | None = None) -> bool:
        """Update EMA weights on every training step (not per epoch)."""
        return step_idx is not None
