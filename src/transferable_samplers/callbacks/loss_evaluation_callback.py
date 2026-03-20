from __future__ import annotations

from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer

from transferable_samplers.utils.dist_utils import get_rank, get_world_size
from transferable_samplers.utils.pylogger import RankedLogger
from transferable_samplers.utils.standardization import standardize_coords
from transferable_samplers.utils.wandb_utils import compute_mean_metrics

logger = RankedLogger(__name__, rank_zero_only=False)


class LossEvaluationCallback(Callback):
    """Evaluate model loss on held-out true samples during validation/test.

    Uses the model's ``compute_primary_loss`` (MSE for flow matching, NLL for
    normalizing flows) without system-size normalization.

    DDP-safe: all ranks compute loss on a shard of the data, then all_reduce to
    get the global mean. Only rank 0 logs metrics.

    Args:
        batch_size: Number of samples per forward pass.
        max_samples: Cap on the number of true samples to evaluate. None = use all.
    """

    def __init__(self, batch_size: int = 256, max_samples: int | None = None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.max_samples = max_samples

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate loss on validation data."""
        self._evaluate(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate loss on test data."""
        self._evaluate(trainer, pl_module, "test")

    def _evaluate(self, trainer: Trainer, pl_module: LightningModule, prefix: str) -> None:
        datamodule = trainer.datamodule
        eval_sequences = datamodule.val_sequences if prefix == "val" else datamodule.test_sequences

        world_size = get_world_size()
        rank = get_rank()

        all_metrics: dict[str, Any] = {}
        for sequence in eval_sequences:
            eval_ctx = datamodule.prepare_eval(sequence=sequence, stage=prefix)

            samples = eval_ctx.true_data.samples
            if self.max_samples is not None:
                samples = samples[: self.max_samples]
            x = standardize_coords(samples, eval_ctx.normalization_std).to(
                device=pl_module.device, dtype=pl_module.dtype
            )

            # Shard samples across ranks (drop remainder for even split)
            n = (x.shape[0] // world_size) * world_size
            x = x[:n]
            chunk_size = n // world_size
            x_local = x[rank * chunk_size : (rank + 1) * chunk_size]

            system_cond = eval_ctx.system_cond

            # Compute loss on local shard
            losses = []
            with torch.no_grad():
                for i in range(0, x_local.shape[0], self.batch_size):
                    x_batch = x_local[i : i + self.batch_size]
                    batch = self._build_batch(x_batch, system_cond, pl_module.device)
                    loss = pl_module.compute_primary_loss(batch).mean()
                    losses.append(loss.detach())

            local_mean = torch.stack(losses).mean()

            # All-reduce to get global mean (each rank has equal-sized shard)
            if world_size > 1:
                torch.distributed.all_reduce(local_mean, op=torch.distributed.ReduceOp.AVG)

            if trainer.is_global_zero:
                key = f"{prefix}/{sequence}/eval-loss"
                value = local_mean.item()
                all_metrics[key] = value
                pl_module.log_dict({key: value})
                logger.info(f"{key}: {value:.6f}")

            # Logging only runs on rank 0 — barrier so all ranks are aligned before next sequence
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            del eval_ctx

        if trainer.is_global_zero:
            mean_metrics = compute_mean_metrics(all_metrics, prefix=prefix)
            pl_module.log_dict(mean_metrics)

        # Logging only runs on rank 0 — barrier so all ranks are aligned before returning
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @staticmethod
    def _build_batch(
        x: torch.Tensor,
        system_cond: Any | None,
        device: torch.device,
    ) -> dict[str, Any]:
        """Build a batch dict from samples and optional system conditioning."""
        batch: dict[str, Any] = {"x": x}
        if system_cond is not None:
            batched = system_cond.for_batch(x.shape[0], device)
            if batched.encodings is not None:
                batch["encodings"] = batched.encodings
            if batched.permutations is not None:
                batch["permutations"] = batched.permutations
        return batch
