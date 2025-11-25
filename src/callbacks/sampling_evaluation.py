import logging
from collections import defaultdict
from statistics import mean as stats_mean, median as stats_median

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from src.evaluation.evaluator import Evaluator
from src.utils.timing_utils import timed_block, timing_metrics

logger = logging.getLogger(__name__)


def detach_and_cpu(obj):
    """
    Recursively detach and move all tensors to CPU within a nested structure.
    Works with dicts, lists, tuples, and tensors.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: detach_and_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_and_cpu(v) for v in obj)
    else:
        return obj  # Leave other data types (int, float, str, etc.) as-is


def add_aggregate_metrics(metrics: dict[str, torch.Tensor], prefix: str = "val") -> dict[str, torch.Tensor]:
    # TODO: add per-length metrics
    """Aggregate metrics across all sequences."""

    mean_dict_list = defaultdict(list)
    median_dict_list = defaultdict(list)
    count_dict = defaultdict(int)

    # Parse and aggregate metrics along peptide sequences
    for key, value in metrics.items():
        if key.startswith(prefix):  # TODO not sure this is needed here
            # Extract sequence and metric name
            parts = key.split("/")
            if len(parts) >= 3:
                metric_name = "/".join(parts[2:])

                # Add to mean and median dictionaries
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

    # Compute mean and median for each metric
    mean_dict = {}
    median_dict = {}
    for key, value in mean_dict_list.items():
        mean_dict[key] = stats_mean(value)

    for key, value in median_dict_list.items():
        median_dict[key] = stats_median(value)

    metrics.update(mean_dict)
    metrics.update(median_dict)
    metrics.update(count_dict)
    return metrics


class SamplingEvaluationCallback(Callback):
    """
    A flexible evaluation callback that runs custom evaluation logic
    at the end of training, validation, or test epochs.
    Supports multi-GPU and works with Lightning's logging system.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        run_on_validation_epoch_end: bool = True,
        run_on_test_epoch_end: bool = False,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.run_on_val = run_on_validation_epoch_end
        self.run_on_test = run_on_test_epoch_end

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.run_on_val:
            self.run_evaluation(stage="val", trainer=trainer, pl_module=pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.run_on_test:
            self.run_evaluation(stage="test", trainer=trainer, pl_module=pl_module)

    def run_evaluation(self, stage: str, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            sequences = trainer.datamodule.test_sequences if stage == "test" else trainer.datamodule.val_sequences

            all_metrics = {}
            all_plots = {}
            
            for sequence in sequences:
                logging.info(f"Evaluating {sequence} on {stage}")

                # Prepare system conditioning and evaluation inputs
                system_cond, evaluation_inputs, target_energy_fn = trainer.datamodule.prepare_eval(sequence, stage)

                # Sample sequences from the model
                samples_data_dict = pl_module.sample_sequence(sequence, system_cond, target_energy_fn)

                if trainer.is_global_zero:
                    # Evaluate the samples
                    metrics, plots = self.evaluator.evaluate(
                        sequence, 
                        samples_data_dict, 
                        evaluation_inputs, 
                        target_energy_fn
                    )

                    for key, value in metrics.items():
                        trainer.log(key, value, sync_dist=True, on_epoch=True)

                    # Merge metrics and plots
                    all_metrics.update(metrics)
                    all_plots.update(plots)

            # Aggregate metrics across sequences
            aggregated_metrics = add_aggregate_metrics(all_metrics, prefix=stage)

            # Log everything in Lightning
            for key, value in aggregated_metrics.items():
                trainer.log(key, value, sync_dist=True, on_epoch=True)

            # Log plots if logger supports it
            if hasattr(trainer.logger, "experiment") and trainer.logger.experiment is not None:
                for plot_name, plot_data in all_plots.items():
                    if isinstance(plot_data, dict):
                        for subplot_name, plot_obj in plot_data.items():
                            if hasattr(trainer.logger.experiment, "log_image"):
                                trainer.logger.experiment.log_image(
                                    f"{stage}/{plot_name}/{subplot_name}", 
                                    plot_obj
                                )
