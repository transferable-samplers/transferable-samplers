from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.utils._pytree as pytree
from lightning import Callback, LightningModule, Trainer

from transferable_samplers.evaluation.diagnostics.smc_plots import plot_smc_diagnostics
from transferable_samplers.evaluation.evaluator import PeptideEnsembleEvaluator
from transferable_samplers.samplers.base_sampler import BaseSampler
from transferable_samplers.utils.pylogger import RankedLogger
from transferable_samplers.utils.wandb_utils import compute_mean_metrics, make_log_image_fn

logger = RankedLogger(__name__, rank_zero_only=False)


class SamplingEvaluationCallback(Callback):
    """Generate conformations and evaluate them at validation/test epoch end.

    All ranks participate in sampling (due to all_gather inside the sampler).
    Evaluation, plotting, saving, and metric logging happen on global rank zero
    only. EMA weight swapping is handled separately by ``EMAWeightAveraging``.

    Args:
        evaluator: Evaluator that computes metrics from generated conformations.
        sampler: Sampler used to generate conformations. If None, evaluation is
            skipped entirely.
        run_diagnostics_kwargs: Extra kwargs forwarded to
            ``pl_module.run_model_diagnostics`` (if the model implements it).
        output_dir: Directory to save ``samples_dict.pt`` and ``diagnostics.pt``.
            Empty string disables saving.
    """

    def __init__(
        self,
        evaluator: PeptideEnsembleEvaluator,
        sampler: BaseSampler | None = None,
        run_diagnostics_kwargs: dict[str, Any] | None = None,
        output_dir: str = "",
    ) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.sampler = sampler
        self.run_diagnostics_kwargs = run_diagnostics_kwargs or {}
        self.output_dir = output_dir

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run evaluation on validation sequences."""
        self.evaluate(trainer, pl_module, "val")
        logger.info("Validation evaluation complete")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run evaluation on test sequences."""
        self.evaluate(trainer, pl_module, "test")
        logger.info("Test evaluation complete")

    def evaluate(self, trainer: Trainer, pl_module: LightningModule, prefix: str) -> None:
        """Sample conformations for each sequence and evaluate on rank zero.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: The LightningModule being trained/evaluated.
            prefix: Either ``"val"`` or ``"test"``, used for metric naming and
                to select which sequences to evaluate.
        """
        if self.sampler is None:
            return

        datamodule = trainer.datamodule
        eval_sequences = datamodule.val_sequences if prefix == "val" else datamodule.test_sequences
        if trainer.sanity_checking:
            eval_sequences = eval_sequences[:1]

        base_log_image_fn = make_log_image_fn(trainer)

        all_metrics: dict[str, Any] = {}
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

            # Only rank 0: evaluate, plot, save, log per-sequence metrics
            if trainer.is_global_zero:
                if diagnostics is not None:
                    plot_smc_diagnostics(diagnostics, log_image_fn)

                self._save_samples(seq_prefix, samples_dict, diagnostics)

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
                    seq_metrics.update(
                        pl_module.run_model_diagnostics(
                            prefix=seq_prefix,
                            system_cond=eval_ctx.system_cond,
                            **self.run_diagnostics_kwargs,
                        )
                    )

                pl_module.log_dict(seq_metrics)
                all_metrics.update(seq_metrics)

            # Evaluator / diagnostics only run on rank 0 — barrier before next sequence so all ranks are aligned
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            del eval_ctx

        if trainer.is_global_zero:
            plt.close("all")
            mean_metrics = compute_mean_metrics(all_metrics, prefix=prefix)
            pl_module.log_dict(mean_metrics)

    def _save_samples(self, seq_prefix: str, samples_dict: dict[str, Any], diagnostics: Any) -> None:
        """Save samples_dict and diagnostics to disk under output_dir/seq_prefix/."""
        if not self.output_dir:
            return

        save_dir = Path(self.output_dir) / seq_prefix
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(samples_dict, str(save_dir / "samples_dict.pt"))
        logger.info(f"Saved samples_dict to {save_dir}/samples_dict.pt")

        if diagnostics is not None:
            torch.save(diagnostics, str(save_dir / "diagnostics.pt"))
            logger.info(f"Saved diagnostics to {save_dir}/diagnostics.pt")
