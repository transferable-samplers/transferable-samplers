from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.utils._pytree as pytree
from lightning import Callback

from src.evaluation.diagnostics.smc_plots import plot_smc_diagnostics
from src.evaluation.evaluator import PeptideEnsembleEvaluator
from src.sampling.base_sampler import BaseSampler
from src.utils.logging_utils import compute_mean_metrics, make_log_image_fn
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class SamplingEvaluationCallback(Callback):
    """Lightning callback that orchestrates sample generation + evaluation.

    On validation/test epoch end:
    1. Loops over sequences, generating samples via self.sampler (all ranks)
    2. Global zero only: evaluates samples, plots, logs per-sequence metrics
    3. Global zero only: aggregates and logs mean metrics across sequences

    EMA weight swapping is handled by the EMAWeightAveraging callback.
    """

    def __init__(
        self,
        evaluator: PeptideEnsembleEvaluator,
        sampler: Optional[BaseSampler] = None,
        run_diagnostics_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.sampler = sampler
        self.run_diagnostics_kwargs = run_diagnostics_kwargs or {}

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

            # ALL ranks must participate in sampling (all_gather inside sampler)
            source_energy = pl_module.build_source_energy(eval_ctx.system_cond)
            samples_dict, diagnostics = self.sampler.sample(
                source_energy,
                eval_ctx.target_energy,
            )

            # Only rank 0: evaluate, plot, log per-sequence metrics
            if trainer.is_global_zero:
                if diagnostics is not None:
                    plot_smc_diagnostics(diagnostics, log_image_fn)

                seq_metrics = self.evaluator.evaluate(
                    samples_dict,
                    eval_ctx,
                    log_image_fn=log_image_fn,
                    prefix=seq_prefix,
                )
                # Had some graph retention issues
                seq_metrics = pytree.tree_map(
                    lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x,
                    seq_metrics,
                )

                if hasattr(pl_module, "run_model_diagnostics"):
                    seq_metrics.update(pl_module.run_model_diagnostics(
                        prefix=seq_prefix,
                        system_cond=eval_ctx.system_cond,
                        **self.run_diagnostics_kwargs,
                    ))

                pl_module.log_dict(seq_metrics)
                all_metrics.update(seq_metrics)

        if trainer.is_global_zero:
            plt.close("all")
            mean_metrics = compute_mean_metrics(all_metrics, prefix=prefix)
            pl_module.log_dict(mean_metrics)
