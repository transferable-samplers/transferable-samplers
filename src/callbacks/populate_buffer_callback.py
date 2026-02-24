from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.utils._pytree as pytree
from lightning import Callback

from src.callbacks.ema_weight_averaging import EMAWeightAveraging
from src.evaluation.evaluator import PeptideEnsembleEvaluator
from src.models.buffer import Buffer
from src.models.samplers.base_sampler import BaseSampler
from src.utils import pylogger
from src.utils.logging_utils import make_log_image_fn

logger = pylogger.RankedLogger(__name__, rank_zero_only=False)


class PopulateBufferCallback(Callback):
    """Lightning callback that owns a sampler and populates the model's buffer for self-improvement training.

    On each training epoch start:
    1. Gets eval context from the datamodule
    2. Builds source energy from the model
    3. Runs the sampler to generate samples
    4. Optionally evaluates the samples
    5. Stores the resampled samples in the model's buffer
    """

    def __init__(self, sampler: BaseSampler, evaluator: Optional[PeptideEnsembleEvaluator] = None):
        super().__init__()
        self.sampler = sampler
        self.evaluator = evaluator

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        assert pl_module.train_from_buffer, (
            "PopulateBufferCallback requires model.train_from_buffer=True."
        )
        assert not self._has_ema_callback(trainer), (
            "EMAWeightAveraging callback should not be used with self-improvement."
        )

        datamodule = trainer.datamodule
        assert datamodule.test_sequences is not None, "Eval sequence name should be set"
        assert len(datamodule.test_sequences) == 1, "Can only self-refine on 1 test sequence at a time."

        sequence = datamodule.test_sequences[0]
        logger.info(f"Generating {self.sampler.num_samples} samples for self-consumption")

        eval_ctx = datamodule.prepare_eval(sequence, stage="test")
        source_energy = pl_module.build_source_energy(eval_ctx.system_cond, use_ema_if_available=True)

        with torch.no_grad():
            samples_dict, diagnostics = self.sampler.sample(
                source_energy, eval_ctx.target_energy
            )

        # Evaluate samples if evaluator is configured
        if self.evaluator is not None and trainer.is_global_zero:
            prefix = f"self_improve/{sequence}"
            base_log_image_fn = make_log_image_fn(trainer)
            log_image_fn = partial(base_log_image_fn, title_prefix=prefix)

            metrics = self.evaluator.evaluate(
                samples_dict,
                eval_ctx,
                log_image_fn=log_image_fn,
                prefix=prefix,
            )
            metrics = pytree.tree_map(
                lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x,
                metrics,
            )
            pl_module.log_dict(metrics)
            plt.close("all")

        batch_transform = getattr(datamodule, "buffer_transforms", None)

        # SMC is much more costly, just assume we want to use it if present.
        key = "smc" if "smc" in samples_dict else "resampled"
        pl_module.set_buffer(Buffer(
            samples=samples_dict[key].samples,
            normalization_std=datamodule.std,
            system_cond=eval_ctx.system_cond,
            batch_transform=batch_transform,
        ))
        logger.info(f"Buffer populated with {len(pl_module._buffer)} resampled samples for sequence '{sequence}'")

    @staticmethod
    def _has_ema_callback(trainer) -> bool:
        """Check if an EMAWeightAveraging callback is present in the trainer."""
        for cb in trainer.callbacks:
            if isinstance(cb, EMAWeightAveraging):
                return True
        return False
