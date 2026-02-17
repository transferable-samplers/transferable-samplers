import inspect
import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Optional

import torch
import torchmetrics
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.neural_networks.ema import EMA
from src.utils.dataclasses import ProposalCond

logger = logging.getLogger(__name__)


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        prior,
        ema_decay: float = 0.0,
        compile: bool = False,
        fix_symmetry: bool = True,
        drop_unfixable_symmetry: bool = False,
        use_distill_loss: bool = False,
        distill_weight: float = 0.5,
        output_dir: str = "",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if args or kwargs:
            logger.warning(f"Unexpected arguments: {args}, {kwargs}")

        self.net = net
        if self.hparams.ema_decay > 0:
            self.net = EMA(net, decay=self.hparams.ema_decay)

        self.prior = prior

        # loss function
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = torchmetrics.MetricCollection({"loss": MeanMetric()}, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.output_dir = output_dir

    @abstractmethod
    def model_step(self, batch) -> torch.Tensor:
        ...

    @abstractmethod
    def sample_proposal(
        self, num_samples: int, proposal_cond: Optional[ProposalCond]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the proposal distribution.

        Args:
            num_samples: Number of samples to generate.
            proposal_cond: Optional conditioning (permutations, encodings).

        Returns:
            (samples, log_q) — samples in normalized space, log_q is the log proposal density.
        """
        ...

    @abstractmethod
    def proposal_energy(self, x: torch.Tensor, proposal_cond: Optional[ProposalCond]) -> torch.Tensor:
        ...

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        assert len(batch["x"].shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"
        loss = self.model_step(batch)
        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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

    @torch.no_grad()
    def eval_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        prefix: str = "val",
    ) -> None:
        loss = self.model_step(batch)
        if prefix == "val":
            self.val_metrics.update(loss)
        elif prefix == "test":
            self.test_metrics.update(loss)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self.eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self.eval_step(batch, batch_idx, prefix="test")

    def on_fit_start(self) -> None:
        if self.hparams.use_distill_loss:
            self.teacher = deepcopy(self.net)

            if self.hparams.ema_decay > 0:
                self.teacher.backup()
                self.teacher.copy_to_model()

            for param in self.teacher.parameters():
                param.requires_grad = False

    def on_train_epoch_start(self) -> None:
        logging.info("Train epoch start")
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        logging.info("Validation epoch start")
        self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        logging.info("Test epoch start")
        self.test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()
        logging.info("Train epoch end")

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()
        logging.info("Validation epoch end")

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()
        logging.info("Test epoch end")

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

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.net, EMA):
            self.net.update_ema()

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("teacher.")}
