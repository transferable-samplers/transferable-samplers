import pytorch_lightning as pl
from typing import Any, Dict


    def add_aggregate_metrics(self, metrics: dict[str, torch.Tensor], prefix: str = "val") -> dict[str, torch.Tensor]:
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
            mean_dict[key] = stats.mean(value)

        for key, value in median_dict_list.items():
            median_dict[key] = stats.median(value)

        metrics.update(mean_dict)
        metrics.update(median_dict)
        metrics.update(count_dict)
        return metrics

    
def detach_and_cpu(
        self, obj
    ):  # TODO hack to have this here? at all? you could just be more careful to detach / cpu?
        """
        Recursively detach and move all tensors to CPU within a nested structure.
        Works with dicts, lists, tuples, and tensors.
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: self.detach_and_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.detach_and_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.detach_and_cpu(v) for v in obj)
        else:
            return obj  # Leave other data types (int, float, str, etc.) as-is




class EvaluationCallback(pl.Callback):
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

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.run_on_val:
            self.run_evaluation(stage="val", trainer=trainer, pl_module=pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.run_on_test:
            self.run_evaluation(stage="test", trainer=trainer, pl_module=pl_module)

    def run_evaluation(self, stage: str, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.eval()
        with torch.no_grad():
            sequences = trainer.datamodule.test_sequences if stage == "test" else trainer.datamodule.val_sequences
            for sequence in sequences:
                model_inputs, evaluation_inputs, energy_fn = trainer.datamodule.prepare_eval(sequence, stage)
                samples_data_dict = pl_module.sample_sequence(sequence, cond_inputs)
                metrics = self.evaluator.evaluate(sequence, samples_data_dict, evaluation_inputs, energy_fn)





            # TODO do the EMA stuff
            result_dict = self.evaluator.evaluate()

        # Log everything in Lightning
        for key, value in result_dict.items():
            trainer.logger.experiment.add_scalar(f"{stage}/{key}", value, trainer.global_step)
            trainer.log(f"{stage}/{key}", value)