import inspect
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any

import torch
import torchmetrics
from lightning import LightningModule
from torchmetrics import MeanMetric

from transferable_samplers.callbacks.ema_weight_averaging import EMAWeightAveraging
from transferable_samplers.models.buffer import Buffer
from transferable_samplers.utils.dataclasses import SourceEnergy, SourceEnergyConfig, SystemCond
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        # pyrefly: ignore [not-a-type]
        scheduler: torch.optim.lr_scheduler,
        prior,
        compile_net: bool = False,
        source_energy_config: SourceEnergyConfig | None = None,
        train_from_buffer: bool = False,
        mean_free_prior: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["source_energy_config"])

        self.net = net
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler
        self.prior = prior
        self.compile_net = compile_net
        self.source_energy_config = source_energy_config
        self.train_from_buffer = train_from_buffer
        self.mean_free_prior = mean_free_prior

        self._buffer = None
        self.train_metrics = torchmetrics.MetricCollection({"loss": MeanMetric()}, prefix="train/")

    @abstractmethod
    # pyrefly: ignore [bad-override]
    def training_step(self, batch, batch_idx: int) -> torch.Tensor: ...

    @abstractmethod
    def generate_proposal(
        self,
        net: torch.nn.Module,
        num_samples: int,
        system_cond: SystemCond | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a single batch of samples from the proposal distribution.

        Args:
            net: The network to use for generation (may be EMA copy).
            num_samples: Number of samples to generate in this batch.
            system_cond: Optional conditioning (permutations, encodings).

        Returns:
            (samples, E_source) — samples in normalized space, E_source is the proposal energy (-logq).
        """
        ...

    @abstractmethod
    def proposal_energy(self, net: torch.nn.Module, x: torch.Tensor, system_cond: SystemCond | None) -> torch.Tensor:
        """Compute proposal energy (-log q) for a single batch.

        Args:
            net: The network to use.
            x: Samples to evaluate (batch, atoms, 3).
            system_cond: Optional conditioning.

        Returns:
            Energy tensor (batch,).
        """
        ...

    def setup(self, stage: str) -> None:
        if self.compile_net and stage == "fit":
            # pyrefly: ignore [bad-assignment]
            self.net = torch.compile(self.net)

        if self.trainer is not None:
            assert self.trainer.limit_val_batches == 1, (
                f"limit_val_batches must be 1 (got {self.trainer.limit_val_batches}). "
                "0 skips validation phases, >1 wastes compute — real evaluation is in callbacks."
            )
            assert self.trainer.limit_test_batches == 1, (
                f"limit_test_batches must be 1 (got {self.trainer.limit_test_batches}). "
                "0 skips test phases, >1 wastes compute — real evaluation is in callbacks."
            )

    # pyrefly: ignore [bad-override]
    def configure_optimizers(self) -> dict[str, Any]:
        # Only parameters with requires_grad=True are passed to optimizer
        # pyrefly: ignore [not-callable]
        optimizer = self.optimizer_fn(params=[p for p in self.parameters() if p.requires_grad])
        if self.scheduler_fn is not None:
            scheduler_fn = self.scheduler_fn
            scheduler_params = inspect.signature(scheduler_fn).parameters

            if "total_steps" in scheduler_params:
                scheduler = scheduler_fn(
                    optimizer=optimizer,
                    total_steps=self.trainer.estimated_stepping_batches,
                )
            else:
                scheduler = scheduler_fn(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return {"optimizer": optimizer}

    def _has_ema_callback(self) -> bool:
        """Check if an EMAWeightAveraging callback is present in the trainer."""
        if self.trainer is None:
            return False
        # pyrefly: ignore [missing-attribute]
        for cb in self.trainer.callbacks:
            if isinstance(cb, EMAWeightAveraging):
                return True
        return False

    def _build_net_copy(self, use_ema_if_available: bool = False) -> torch.nn.Module:
        """Return a detached copy of the EMA-averaged net (or plain net if no EMA callback)."""
        if use_ema_if_available:
            # pyrefly: ignore [missing-attribute]
            for cb in self.trainer.callbacks:
                if isinstance(cb, EMAWeightAveraging):
                    # pyrefly: ignore [missing-attribute]
                    return deepcopy(cb._average_model.module.net)
        return deepcopy(self.net)

    def build_source_energy(
        self,
        system_cond: SystemCond | None,
        use_ema_if_available: bool = False,
    ) -> SourceEnergy:
        """Build a SourceEnergy with batched sample and energy callables.

        Args:
            system_cond: Optional conditioning (permutations, encodings).
            use_ema_if_available: If True uses the EMA weights if present.
        """
        assert self.source_energy_config is not None, "source_energy_config must be set to build SourceEnergy."
        net = self._build_net_copy(use_ema_if_available=use_ema_if_available)

        return SourceEnergy(
            sample_fn=partial(self.generate_proposal, net, system_cond=system_cond),
            energy_fn=partial(self.proposal_energy, net, system_cond=system_cond),
            sample_batch_size=self.source_energy_config.sample_batch_size,
            energy_batch_size=self.source_energy_config.energy_batch_size,
            grad_batch_size=self.source_energy_config.grad_batch_size,
            use_com_adjustment=self.source_energy_config.use_com_adjustment,
        )

    def set_buffer(self, buffer: Buffer) -> None:
        """Set the sample buffer for self-improvement training."""
        self._buffer = buffer

    def on_train_epoch_start(self) -> None:
        logger.info("Train epoch start")
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()
        logger.info("Train epoch end")

    def on_before_optimizer_step(self, optimizer, *args, **kwargs) -> None:
        total_norm = 0.0
        for param in self.trainer.lightning_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.log_dict({"train/grad_norm": total_norm}, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        logger.info("Validation epoch start")

    def on_validation_epoch_end(self) -> None:
        logger.info("Validation epoch end")

    # pyrefly: ignore [bad-override]
    def validation_step(self, batch, batch_idx):
        "NOTE: these only exist for Lightning compatibility. All evaluation is handled by custom callbacks."
        return None

    def on_test_epoch_start(self) -> None:
        logger.info("Test epoch start")

    def on_test_epoch_end(self) -> None:
        logger.info("Test epoch end")

    # pyrefly: ignore [bad-override]
    def test_step(self, batch, batch_idx):
        "NOTE: these only exist for Lightning compatibility. All evaluation is handled by custom callbacks."
        return None
