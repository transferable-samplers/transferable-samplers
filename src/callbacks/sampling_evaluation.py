import logging
import statistics as stats
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger

from src.evaluation.evaluator import Evaluator
from src.models.neural_networks.ema import EMA

logger = logging.getLogger(__name__)


def detach_and_cpu(obj):
    """Recursively detach and move all tensors to CPU within a nested structure."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: detach_and_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_and_cpu(v) for v in obj)
    else:
        return obj


def add_aggregate_metrics(metrics: dict, prefix: str = "val") -> dict:
    """Aggregate metrics across all sequences by computing mean and median."""
    mean_dict_list = defaultdict(list)
    median_dict_list = defaultdict(list)
    count_dict = defaultdict(int)

    for key, value in metrics.items():
        if key.startswith(prefix):
            parts = key.split("/")
            metric_name = "/".join(parts[2:])

            mean_key = f"{prefix}/mean/{metric_name}"
            median_key = f"{prefix}/median/{metric_name}"
            count_key = f"{prefix}/count/{metric_name}"

            if isinstance(value, torch.Tensor):
                value = value.item()
            elif isinstance(value, (int, float)):
                value = float(value)

            mean_dict_list[mean_key].append(value)
            median_dict_list[median_key].append(value)
            count_dict[count_key] += 1

    mean_dict = {k: stats.mean(v) for k, v in mean_dict_list.items()}
    median_dict = {k: stats.median(v) for k, v in median_dict_list.items()}

    metrics.update(mean_dict)
    metrics.update(median_dict)
    metrics.update(count_dict)
    return metrics


class SamplingEvaluationCallback(Callback):
    """Lightning callback that orchestrates sample generation + evaluation.

    Replaces the evaluate_all / evaluate / on_eval_epoch_end logic that
    previously lived in TransferableBoltzmannGeneratorLitModule.

    On validation/test epoch end:
    1. Swaps to EMA weights if applicable
    2. Loops over sequences, generating samples via model.sampler
    3. Evaluates samples via the Evaluator
    4. Aggregates metrics across sequences
    5. Handles DDP broadcasting of metrics
    6. Logs everything
    """

    def __init__(self, evaluator: Evaluator):
        super().__init__()
        self.evaluator = evaluator

    def on_validation_epoch_end(self, trainer, pl_module):
        self._evaluate_epoch(trainer, pl_module, "val")
        logger.info("Validation evaluation complete")

    def on_test_epoch_end(self, trainer, pl_module):
        self._evaluate_epoch(trainer, pl_module, "test")
        logger.info("Test evaluation complete")

    def _evaluate_epoch(self, trainer, pl_module, prefix):
        """Run evaluation for all sequences in a given stage."""
        if pl_module.sampler is None:
            return

        use_ema = isinstance(pl_module.net, EMA) and pl_module.hparams.ema_decay > 0
        if use_ema:
            pl_module.net.backup()
            pl_module.net.copy_to_model()

        try:
            self._evaluate_all(trainer, pl_module, prefix)
        finally:
            if use_ema:
                pl_module.net.restore_to_model()
            plt.close("all")

    def _evaluate_all(self, trainer, pl_module, prefix):
        """Loop over sequences, generate samples, evaluate, aggregate, and log."""
        datamodule = trainer.datamodule
        eval_sequences = datamodule.val_sequences if prefix == "val" else datamodule.test_sequences

        metrics = {}
        for sequence in eval_sequences:
            eval_ctx = datamodule.prepare_eval(sequence=sequence, stage=prefix)
            logger.info(f"Evaluating {sequence} samples")

            # Generate samples via the sampler
            samples_dict = pl_module.sampler.sample(
                pl_module,
                eval_ctx.proposal_cond,
                eval_ctx.target_energy_fn,
                prefix=f"{prefix}/{sequence}",
            )

            # Build log_image_fn from the model's loggers
            log_image_fn = self._make_log_image_fn(pl_module)

            # Evaluate: chirality fixing, metrics, plots
            seq_metrics = self.evaluator.evaluate(
                sequence,
                samples_dict,
                eval_ctx,
                log_image_fn=log_image_fn,
                prefix=f"{prefix}/{sequence}",
            )

            metrics.update(seq_metrics)

        # Aggregate metrics across all sequences
        if pl_module.local_rank == 0:
            metrics = detach_and_cpu(metrics)
            metric_object_list = [add_aggregate_metrics(metrics, prefix=prefix)]
        else:
            metric_object_list = [None]

        if trainer.world_size > 1:
            torch.distributed.broadcast_object_list(metric_object_list, src=0)

        pl_module.log_dict(metric_object_list[0])

    @staticmethod
    def _make_log_image_fn(pl_module):
        """Create an image logging function from the model's loggers."""
        def log_image(img, title=None):
            if pl_module.loggers is not None:
                for lg in pl_module.loggers:
                    if isinstance(lg, WandbLogger):
                        lg.log_image(title, [img])
        return log_image
