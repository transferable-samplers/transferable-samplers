import logging

import torch
from lightning import LightningModule

from src.models.neural_networks.ema import EMA

logger = logging.getLogger(__name__)


class BaseLightningModule(LightningModule):
    """
    Base Lightning module with common functionality for gradient monitoring,
    EMA handling, and state dict filtering.
    """

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
                f"Generating {self.hparams.sampling_config.num_self_improve_proposal_samples} Samples"
                " for self-consumption"
            )
            self.net.eval()

            if self.hparams.ema_decay > 0:
                self.net.backup()
                self.net.copy_to_model()

                with torch.no_grad():
                    samples = self.generate_and_resample(
                        num_proposal_samples=self.hparams.sampling_config.num_self_improve_proposal_samples,
                    )

                self.net.restore_to_model()

            else:
                with torch.no_grad():
                    samples = self.generate_and_resample(
                        num_proposal_samples=self.hparams.sampling_config.num_self_improve_proposal_samples,
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
