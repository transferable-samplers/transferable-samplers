import inspect
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any, Optional

import torch
import torchmetrics
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.callbacks.ema_weight_averaging import EMAWeightAveraging
from src.models.buffer import Buffer
from src.models.samplers.base_sampler import BaseSampler
from src.utils import pylogger
from src.utils.dataclasses import DistOps, ProposalModel, SystemCond

logger = pylogger.RankedLogger(__name__, rank_zero_only=False)


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        prior,
        compile: bool = False,
        fix_symmetry: bool = True,
        drop_unfixable_symmetry: bool = False,
        use_distill_loss: bool = False,
        distill_weight: float = 0.5,
        output_dir: str = "",
        sampler: Optional[BaseSampler] = None,
        train_from_buffer: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["sampler"])
        if args or kwargs:
            logger.warning(f"Unexpected arguments: {args}, {kwargs}")

        self.net = net

        self.prior = prior

        self.sampler = sampler
        self.train_from_buffer = train_from_buffer
        self._buffer: Optional[Buffer] = None

        self.train_metrics = torchmetrics.MetricCollection({"loss": MeanMetric()}, prefix="train/")

        self.output_dir = output_dir

        if self.hparams.use_distill_loss:
            logger.info("Using distillation loss with weight {:.3f}".format(self.hparams.distill_weight))
            logger.info("Copying net to teacher for distillation loss")
            self.teacher = deepcopy(self.net)
            for param in self.teacher.parameters():
                param.requires_grad_(False)
            self.teacher.eval()

    @abstractmethod
    def sample_proposal(
        self, net: torch.nn.Module, num_samples: int, system_cond: Optional[SystemCond], log_metrics: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the proposal distribution.

        Args:
            net: The network to use for generation (may be EMA copy).
            num_samples: Number of samples to generate.
            system_cond: Optional conditioning (permutations, encodings).
            log_metrics: Whether to log metrics via self.log().

        Returns:
            (samples, log_q) — samples in normalized space, log_q is the log proposal density.
        """
        ...

    @abstractmethod
    def proposal_energy(self, net: torch.nn.Module, x: torch.Tensor, system_cond: Optional[SystemCond]) -> torch.Tensor:
        ...

    @abstractmethod
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        ...

    def build_proposal_model(self, system_cond: Optional[SystemCond], use_ema_if_available: bool = False) -> ProposalModel:
        """Build a ProposalModel with system_cond and net pre-bound.

        Args:
            system_cond: Optional conditioning (permutations, encodings).
            use_ema_if_available: If True uses the EMA weights if present.
        """
        net = self._build_net_copy(use_ema_if_available=use_ema_if_available)
        return ProposalModel(
            sample_proposal=partial(self.sample_proposal, net, system_cond=system_cond),
            proposal_energy=partial(self.proposal_energy, net, system_cond=system_cond),
        )

    def build_dist_ops(self) -> DistOps:
        """Build a DistOps with the current trainer's distributed state."""
        return DistOps(
            world_size=self.trainer.world_size if self.trainer else 1,
            local_rank=self.local_rank if self.trainer else 0,
            all_gather=self.all_gather,
        )

    def sample(self, system_cond, target_energy_fn, log_metrics: bool = True):
        """Run the sampler. Delegates to self.sampler.sample()."""
        assert self.sampler is not None, "No sampler configured on this model."
        proposal_model = self.build_proposal_model(system_cond)
        dist_ops = self.build_dist_ops()
        return self.sampler.sample(proposal_model, target_energy_fn, dist_ops=dist_ops, log_metrics=log_metrics)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
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

    def configure_optimizers(self) -> dict[str, Any]:
        # Only parameters with requires_grad=True are passed to optimizer (e.g not self.teacher)
        optimizer = self.hparams.optimizer(params = [p for p in self.parameters() if p.requires_grad])
        if self.hparams.scheduler is not None:
            scheduler_fn = self.hparams.scheduler
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

    def validation_step(self, batch, batch_idx):
        "NOTE: these only exist for Lightning compatibility. All evaluation is handled by custom callbacks."
        return None

    def test_step(self, batch, batch_idx):
        "NOTE: these only exist for Lightning compatibility. All evaluation is handled by custom callbacks."
        return None

    def _has_ema_callback(self) -> bool:
        """Check if an EMAWeightAveraging callback is present in the trainer."""
        if self.trainer is None:
            return False
        for cb in self.trainer.callbacks:
            if isinstance(cb, EMAWeightAveraging):
                return True
        return False

    def _build_net_copy(self, use_ema_if_available: bool = False) -> torch.nn.Module:
        """Return a detached copy of the EMA-averaged net (or plain net if no EMA callback)."""
        if use_ema_if_available:
            for cb in self.trainer.callbacks:
                if isinstance(cb, EMAWeightAveraging):
                    return deepcopy(cb._average_model.module.net)
        return deepcopy(self.net)

    def on_train_epoch_start(self) -> None:
        logger.info("Train epoch start")
        self.train_metrics.reset()

        if self.hparams.use_distill_loss:
            assert not self._has_ema_callback(), "EMAWeightAveraging callback should not be used with distillation loss."

        if self.train_from_buffer:
            assert not self._has_ema_callback(), "EMAWeightAveraging callback should not be used with train_from_buffer."
            assert self.sampler is not None, "train_from_buffer requires a sampler on the model."
            self._populate_buffer()

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()
        logger.info("Train epoch end")

    def on_validation_epoch_start(self) -> None:
        logger.info("Validation epoch start")

    def on_validation_epoch_end(self) -> None:
        logger.info("Validation epoch end")

    def on_test_epoch_start(self) -> None:
        logger.info("Test epoch start")

    def on_test_epoch_end(self) -> None:
        logger.info("Test epoch end")

    def on_after_backward(self) -> None:
        valid_gradients = True
        flat_grads = torch.cat([p.grad.view(-1) for p in self.parameters() if p.grad is not None])
        global_norm = torch.norm(flat_grads, p=2)
        for _name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())

                if not valid_gradients:
                    break

        self.log("global_gradient_norm", global_norm, on_step=True, prog_bar=True)
        if not valid_gradients:
            logger.warning("detected inf or nan values in gradients. not updating model parameters")
            self.zero_grad()
            return

    def on_before_optimizer_step(self, optimizer, *args, **kwargs) -> None:
        total_norm = 0.0
        for param in self.trainer.lightning_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.log_dict({"train/grad_norm": total_norm}, prog_bar=True)

    def _populate_buffer(self):
        """Generate new samples and store in the buffer for self-improvement training."""
        datamodule = self.trainer.datamodule
        assert datamodule.test_sequences is not None, "Eval sequence name should be set"
        assert len(datamodule.test_sequences) == 1, "Can only self-refine on 1 test sequence at a time."

        sequence = datamodule.test_sequences[0]
        logger.info(f"Generating {self.sampler.num_samples} samples for self-consumption")

        eval_ctx = datamodule.prepare_eval(sequence, stage="test")
        proposal_model = self.build_proposal_model(eval_ctx.system_cond, use_ema_if_available=True)
        dist_ops = self.build_dist_ops()

        with torch.no_grad():
            samples_dict = self.sampler.sample(
                proposal_model, eval_ctx.target_energy_fn, dist_ops=dist_ops, log_metrics=False
            )

        batch_transform = getattr(datamodule, "buffer_transforms", None)

        self._buffer = Buffer(
            samples=samples_dict["resampled"].samples,
            normalization_std=datamodule.std,
            system_cond=eval_ctx.system_cond,
            batch_transform=batch_transform,
        )
        logger.info(f"Buffer populated with {len(self._buffer)} resampled samples for sequence '{sequence}'")