import inspect
import logging
import os
from copy import deepcopy
from typing import Any, Optional

import torch
import torchmetrics
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.models.neural_networks.ema import EMA
from src.models.samplers.sampler import Sampler
from src.models.priors.prior import Prior
from src.utils.dataclasses import ProposalModel, SamplesData


logger = logging.getLogger(__name__)


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: LightningDataModule,
        prior: Prior,
        sampler: Sampler,
        ema_decay: float,
        compile: bool,
    ) -> None:
        """
        Initialize a `BaseLightningModule`.
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param datamodule: The data module to use for training.
        :param sampler: The sampler to use for sampling.
        :param ema_decay: The decay rate for the EMA.
        :param compile: Whether to compile the model.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=("datamodule"))

        self.net = net
        # TODO - I think it's safer to just always use EMA model but with zero decay
        if self.hparams.ema_decay > 0:
            self.net = EMA(net, decay=self.hparams.ema_decay)

        self.datamodule = datamodule

        self.prior = prior

        self.sampler = sampler
        if self.sampler is not None:
            self.sampler.log_image_fn = self.log_image_fn

        # loss function
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = torchmetrics.MetricCollection({"loss": MeanMetric()}, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.eval_encodings = None
        self.eval_energy = None

    def log_image_fn(self, img: torch.Tensor, title: str = None) -> None:
        """Log an image to the logger.

        :param img: The image to log.
        :param title: The title of the image.
        """
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image(title, [img])

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        assert len(batch["x"].shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"
        loss = self.model_step(batch)
        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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

    def predict_step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of samples.

        :param batch: A batch of (dummy) data.
        :return: A tuple containing the generated samples, the log probability, and the prior
            samples.
        """
        samples, log_p, prior_samples = self.batched_generate_samples(batch.shape[0])
        return samples, log_p, prior_samples


    def on_after_backward(self) -> None:
        """Validate gradients and log gradient norms after backward pass."""
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

    # https://github.com/Lightning-AI/pytorch-lightning/issues/1462
    def on_before_optimizer_step(self, optimizer, *args, **kwargs) -> None:
        """Log gradient norm before optimizer step."""
        total_norm = 0.0
        for param in self.trainer.lightning_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.log_dict({"train/grad_norm": total_norm}, prog_bar=True)

    def optimizer_step(self, *args, **kwargs):
        """Perform optimizer step and update EMA if applicable."""
        super().optimizer_step(*args, **kwargs)
        if hasattr(self, "net") and isinstance(self.net, EMA):
            self.net.update_ema()

    def on_fit_start(self) -> None:
        """
        Called at the very beginning of the fit loop, after setup() has been called.
        Here we make a copy of the model to serve as a teacher during self-refinement.
        This ensures the "student" model does not drift to far from the initial model (teacher).
        """

        if self.hparams.use_distill_loss and self.hparams.self_improve:
            self.teacher = deepcopy(self.net)

            # ema params as teacher model if available
            if self.hparams.ema_decay > 0:
                self.teacher.backup()
                self.teacher.copy_to_model()

            # Freeze the teacher network's parameters
            for param in self.teacher.parameters():
                param.requires_grad = False

    def on_train_epoch_start(self) -> None:
        logging.info("Train epoch start")

        # if doing self-refinement: generate and reweight samples
        # at start of every epoch to finetune the model
        if self.hparams.get("self_improve", False):
            logging.info(
                f"Generating {self.hparams.proposal_config.num_self_improve_proposal_samples} Samples"
                " for self-consumption"
            )
            self.net.eval()

            if self.hparams.ema_decay > 0:
                self.net.backup()
                self.net.copy_to_model()

                with torch.no_grad():
                    samples = self.generate_and_resample(
                        num_proposal_samples=self.hparams.proposal_config.num_self_improve_proposal_samples,
                    )

                self.net.restore_to_model()

            else:
                with torch.no_grad():
                    samples = self.generate_and_resample(
                        num_proposal_samples=self.hparams.proposal_config.num_self_improve_proposal_samples,
                    )

            self.net.train()

            # add the IS samples into the buffer
            self.datamodule.data_train.buffer.add(samples, self.datamodule.test_sequences[0])

            # save the buffer into memory
            self.datamodule.save_buffer()

        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()
        logging.info("Train epoch end")

    def on_validation_epoch_start(self) -> None:
        logging.info("Validation epoch start")
        self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        logging.info("Test epoch start")
        self.test_metrics.reset()

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end(self.val_metrics, "val")
        logging.info("Validation epoch end")

    def on_test_epoch_end(self) -> None:
        self.on_eval_epoch_end(self.test_metrics, "test")
        logging.info("Test epoch end")

    def state_dict(self, *args, **kwargs):
        """Filter out teacher parameters from state dict."""
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("teacher.")}


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
        """Perform a single validation step on a batch of data from the validation set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="test")

    def batched_sample_proposal(
        self,
        num_samples: int,
        batch_size: int,
        system_cond: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proposal_sample_batches = []
        log_q_theta_batches = []
        prior_sample_batches = []
        for _ in tqdm(range(num_samples // batch_size)):
            proposal_samples, log_q_theta, prior_samples = self.generate_samples(
                batch_size, system_cond=system_cond
            )
            proposal_sample_batches.append(proposal_samples)
            log_q_theta_batches.append(log_q_theta)
            prior_sample_batches.append(prior_samples)
        if num_samples % batch_size > 0:
            s, lp, ps = self.generate_samples(
                num_samples % batch_size, system_cond=system_cond
            )
            proposal_sample_batches.append(s)
            log_q_theta_batches.append(lp)
            prior_sample_batches.append(ps)
        proposal_samples = torch.cat(proposal_sample_batches, dim=0)
        log_q_theta = torch.cat(log_q_theta_batches, dim=0)
        prior_samples = torch.cat(prior_sample_batches, dim=0)
        return proposal_samples, log_q_theta, prior_samples

    def on_eval_epoch_end(self, metrics, prefix: str = "val") -> None:
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()
        # Evaluation is now handled by EvaluationCallback

    def get_proposal_model(self) -> ProposalModel:
        return ProposalModel(
            sample_fn=self.sample_proposal,
            energy_fn=self.proposal_energy_fn,
        )

    def sample_proposal(
        self, batch_size: int, system_cond: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """
        raise NotImplementedError

    def proposal_energy_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Compute proposal energy for a given input.

        :param x: Input tensor representing samples from the proposal distribution.
        :return: Energy values for the proposal samples.
        """
        raise NotImplementedError
