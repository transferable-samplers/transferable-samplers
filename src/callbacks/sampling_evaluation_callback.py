import inspect
from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.utils._pytree as pytree
from lightning import Callback

from src.evaluation.evaluator import Evaluator
from src.models.samplers.base_sampler import BaseSampler
from src.utils import pylogger
from src.utils.logging_utils import compute_mean_metrics, make_log_image_fn

logger = pylogger.RankedLogger(__name__, rank_zero_only=False)


class SamplingEvaluationCallback(Callback):
    """Lightning callback that orchestrates sample generation + evaluation.

    On validation/test epoch end:
    1. Loops over sequences, generating samples via self.sampler (all ranks)
    2. Global zero only: evaluates samples, plots, logs per-sequence metrics
    3. Global zero only: aggregates and logs mean metrics across sequences

    EMA weight swapping is handled by the EMAWeightAveraging callback.
    """

    def __init__(self, evaluator: Evaluator, sampler: Optional[BaseSampler] = None):
        super().__init__()
        self.evaluator = evaluator
        self.sampler = sampler

    def on_validation_epoch_end(self, trainer, pl_module):
        self.evaluate(trainer, pl_module, "val")
        logger.info("Validation evaluation complete")

    def on_test_epoch_end(self, trainer, pl_module):
        self.evaluate(trainer, pl_module, "test")
        logger.info("Test evaluation complete")

    def evaluate(self, trainer, pl_module, prefix):
        if self.sampler is None:
            return

        datamodule = trainer.datamodule
        eval_sequences = datamodule.val_sequences if prefix == "val" else datamodule.test_sequences

        base_log_image_fn = make_log_image_fn(trainer)

        all_metrics = {}
        for sequence in eval_sequences:
            eval_ctx = datamodule.prepare_eval(sequence=sequence, stage=prefix)
            logger.info(f"Evaluating {sequence} samples")

            seq_prefix = f"{prefix}/{sequence}"
            log_image_fn = partial(base_log_image_fn, title_prefix=seq_prefix)

            # ALL ranks must participate in sampling (all_gather)
            sample_kwargs = {}
            if "log_image_fn" in inspect.signature(self.sampler.sample).parameters:
                sample_kwargs["log_image_fn"] = log_image_fn
            samples_dict = self.sampler.sample(
                pl_module,
                eval_ctx.proposal_cond,
                eval_ctx.target_energy_fn,
                **sample_kwargs,
            )

            # Only rank 0: evaluate, plot, log per-sequence metrics
            if trainer.is_global_zero:
                seq_metrics = self.evaluator.evaluate(
                    sequence,
                    samples_dict,
                    eval_ctx,
                    log_image_fn=log_image_fn,
                    prefix=seq_prefix,
                    normalization_std=datamodule.std,
                )
                # Had some graph retention issues
                seq_metrics = pytree.tree_map(
                    lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x,
                    seq_metrics,
                )
                pl_module.log_dict(seq_metrics)
                all_metrics.update(seq_metrics)

        if trainer.is_global_zero:
            plt.close("all")
            mean_metrics = compute_mean_metrics(all_metrics, prefix=prefix)
            pl_module.log_dict(mean_metrics)
