import inspect
import logging
import os
import statistics as stats
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional

import hydra
import matplotlib.pyplot as plt
import torch
import torchmetrics
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric
from tqdm import tqdm, trange

from src.models.neural_networks.ema import EMA
from src.models.priors import NormalDistribution
from src.models.samplers.base_sampler import SMCSampler
from src.models.utils import get_symmetry_change, resample
from src.utils.data_types import SamplesData

logger = logging.getLogger(__name__)


class TransferableBoltzmannGeneratorLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: LightningDataModule,
        smc_sampler: SMCSampler,
        sampling_config: dict,
        ema_decay: float,
        compile: bool,
        use_com_adjustment: bool = False,
        fix_symmetry: bool = True,
        drop_unfixable_symmetry: bool = False,
        use_distill_loss: bool = False,
        distill_weight: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `FlowMatchLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=("datamodule"))
        if args or kwargs:
            logger.warning(f"Unexpected arguments: {args}, {kwargs}")

        self.net = net
        if self.hparams.ema_decay > 0:
            self.net = EMA(net, decay=self.hparams.ema_decay)

        self.datamodule = datamodule

        self.smc_sampler = None
        if smc_sampler is not None:
            self.smc_sampler = smc_sampler(
                log_image_fn=self.log_image,
            )
        
        # loss function
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = torchmetrics.MetricCollection({"loss": MeanMetric()}, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.prior = NormalDistribution(
            self.datamodule.hparams.num_dimensions,  # for transferable this will be the dim of the largest peptide
            mean_free=self.hparams.mean_free_prior,
        )

    def log_image(self, img: torch.Tensor, title: str = None) -> None:
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

    def batched_generate_samples(
        self,
        total_size: int,
        permutations: Optional[dict[str, torch.Tensor]] = None,
        encodings: Optional[dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        dummy_ll: bool = False,
        log_invert_error: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size is None:
            batch_size = self.hparams.sampling_config.batch_size
        samples = []
        log_ps = []
        prior_samples = []
        for _ in tqdm(range(total_size // batch_size)):
            s, lp, ps = self.generate_samples(
                batch_size, permutations=permutations, encodings=encodings, dummy_ll=dummy_ll
            )
            samples.append(s)
            log_ps.append(lp)
            prior_samples.append(ps)
        if total_size % batch_size > 0:
            s, lp, ps = self.generate_samples(
                total_size % batch_size, permutations=permutations, encodings=encodings, dummy_ll=dummy_ll
            )
            samples.append(s)
            log_ps.append(lp)
            prior_samples.append(ps)
        samples = torch.cat(samples, dim=0)
        log_ps = torch.cat(log_ps, dim=0)
        prior_samples = torch.cat(prior_samples, dim=0)
        return samples, log_ps, prior_samples

    def generate_samples(
        self, batch_size: int, encodings: Optional[dict[str, torch.Tensor]] = None, n_timesteps: int = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """
        raise NotImplementedError

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

    def on_eval_epoch_end(self, metrics, prefix: str = "val") -> None:
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()
        if self.hparams.ema_decay > 0:
            self.net.backup()
            self.net.copy_to_model()
            self.evaluate_all(prefix)
            self.net.restore_to_model()
        else:
            self.evaluate_all(prefix)
        plt.close("all")

    def add_aggregate_metrics(self, metrics: dict[str, torch.Tensor], prefix: str = "val") -> dict[str, torch.Tensor]:
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

    def evaluate_all(self, prefix):
        metrics = {}
        eval_sequences = self.datamodule.val_sequences if prefix.startswith("val") else self.datamodule.test_sequences
        for sequence in eval_sequences:
            # TODO: single peptides expects prefix as input while transferable expects sequence as input
            true_samples, permutations, encodings, energy_fn, tica_model = self.datamodule.prepare_eval(
                prefix=prefix, sequence=sequence
            )
            logging.info(f"Evaluating {sequence} samples")
            metrics.update(
                self.evaluate(
                    sequence,
                    true_samples,
                    permutations,
                    encodings,
                    energy_fn,
                    tica_model,
                    prefix=f"{prefix}/{sequence}",
                    proposal_generator=self.batched_generate_samples,
                )
            )

        # Aggregate metrics across all sequences
        if self.local_rank == 0:
            metrics = self.detach_and_cpu(metrics)  # Ensure all tensors are detached and on CPU
            metric_object_list = [self.add_aggregate_metrics(metrics, prefix=prefix)]
        else:
            metric_object_list = [None]  # List must have same length for broadcast
        if self.trainer.world_size > 1:
            # Broadcast metrics to all processes - must log from all for checkpointing
            torch.distributed.broadcast_object_list(metric_object_list, src=0)
        self.log_dict(metric_object_list[0])

    @torch.no_grad()
    def evaluate(
        self,
        sequence,
        true_samples,
        permutations,
        encodings,
        energy_fn,
        tica_model=None,
        prefix: str = "val",
        proposal_generator=None,
        output_dir=None,
    ) -> None:
        """Generates samples from the proposal and runs SMC if enabled.
        Also computes metrics, through the datamodule function "metrics_and_plots".
        """

        metrics = {}

        true_data = SamplesData(
            self.datamodule.unnormalize(true_samples),
            energy_fn(true_samples),
        )

        # Define proposal generator
        if proposal_generator is None:
            proposal_generator = self.batched_generate_samples
            if "dummy_ll" in self.hparams and self.hparams.dummy_ll:
                proposal_generator = lambda x: self.batched_generate_samples(x, dummy_ll=True)

        if prefix.startswith("test"):
            num_proposal_samples = self.hparams.sampling_config.num_test_proposal_samples
        else:
            num_proposal_samples = self.hparams.sampling_config.num_proposal_samples

        if self.hparams.sampling_config.get("load_samples_path", None) is None:
            # Generate samples and record time
            torch.cuda.synchronize()
            start_time = time.time()
            proposal_samples, proposal_log_q, prior_samples = proposal_generator(
                num_proposal_samples, permutations, encodings
            )
            torch.cuda.synchronize()
            time_duration = time.time() - start_time

            metrics.update(
                {
                    f"{prefix}/samples_walltime": time_duration,
                    f"{prefix}/samples_per_second": len(proposal_samples) / time_duration,
                    f"{prefix}/seconds_per_sample": time_duration / len(proposal_samples),
                }
            )

            # Save samples to disk
            samples_dict = {
                "prior_samples": prior_samples,
                "proposal_samples": proposal_samples,
                "proposal_log_q": proposal_log_q,
            }
            if output_dir is None:
                output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            if self.local_rank == 0:
                os.makedirs(f"{output_dir}/{prefix}", exist_ok=True)
                if self.hparams.sampling_config.get("subset_idx") is not None:
                    torch.save(
                        samples_dict, f"{output_dir}/{prefix}/samples_{self.hparams.sampling_config.subset_idx}.pt"
                    )
                    logging.info(
                        f"Saving {len(proposal_samples)} samples to {output_dir} "
                        "/{prefix}/samples_{self.hparams.sampling_config.subset_idx}.pt"
                    )
                    return {}  # early return if subset_idx is set - need to post-process these samples in notebook
                else:
                    torch.save(samples_dict, f"{output_dir}/{prefix}/samples.pt")
                    logging.info(f"Saving {len(proposal_samples)} samples to {output_dir}/{prefix}/samples.pt")
        else:
            # Load samples from disk
            samples_path = self.hparams.sampling_config.load_samples_path
            logging.info(f"Loading proposal samples from {samples_path}")
            samples_dict = torch.load(samples_path, map_location=self.device)
            proposal_samples = samples_dict["proposal_samples"]
            proposal_log_q = samples_dict["proposal_log_q"]
            prior_samples = samples_dict["prior_samples"]
            logging.info(f"Loaded {len(proposal_samples)} samples")

        # Compute energy
        proposal_samples_energy = energy_fn(proposal_samples)

        # Datatype for easier metrics and plotting
        proposal_data = SamplesData(
            self.datamodule.unnormalize(proposal_samples),
            proposal_samples_energy,
        )

        # Compute proposal center of mass std
        coms = proposal_samples.mean(dim=1, keepdim=False)
        proposal_com_std = coms.std()
        # TODO little scary relying on this class attribute! - gets used in self.proposal_energy
        # when use_com_adjustment=True
        self.proposal_com_std = proposal_com_std
        logging.info(f"Proposal CoM std: {proposal_com_std}")
        self.log(f"{prefix}/proposal_com_std", proposal_com_std, sync_dist=True)

        temp_proposal_samples = proposal_samples.clone()

        first_symmetry_change = get_symmetry_change(
            self.datamodule.unnormalize(true_samples),
            self.datamodule.unnormalize(temp_proposal_samples),
            self.datamodule.topology_dict[sequence],
        )

        correct_symmetry_rate = 1 - first_symmetry_change.float().mean().item()

        temp_proposal_samples[first_symmetry_change] *= -1

        second_symmetry_change = get_symmetry_change(
            self.datamodule.unnormalize(true_samples),
            self.datamodule.unnormalize(temp_proposal_samples),
            self.datamodule.topology_dict[sequence],
        )

        uncorrectable_symmetry_rate = second_symmetry_change.float().mean().item()

        if self.hparams.fix_symmetry:
            proposal_samples[first_symmetry_change] *= -1

            if self.hparams.drop_unfixable_symmetry:  # only makes sense to drop if symmetry is fixed
                proposal_samples = proposal_samples[~second_symmetry_change]
                proposal_log_q = proposal_log_q[~second_symmetry_change]
                proposal_samples_energy = proposal_samples_energy[~second_symmetry_change]

        metrics.update(
            {
                f"{prefix}/proposal/correct_symmetry_rate": correct_symmetry_rate,
                f"{prefix}/proposal/uncorrectable_symmetry_rate": uncorrectable_symmetry_rate,
            }
        )

        # Datatype for easier metrics and plotting
        proposal_data = SamplesData(
            self.datamodule.unnormalize(proposal_samples),
            proposal_samples_energy,
        )

        # Apply CoM adjustment to energy, this must be done here for compatibility with CNFs
        if self.hparams.sampling_config.get("use_com_adjustment", False):
            logging.info("Applying center of mass energy adjustment")
            proposal_log_q = proposal_log_q + self.com_energy_adjustment(proposal_samples)

        # Compute resampling index
        # proposal_log_p - proposal_log_q
        resampling_logits = -proposal_samples_energy - proposal_log_q

        # Filter samples based on logit clipping - this affects both IS and SMC
        if self.hparams.sampling_config.clip_reweighting_logits:
            clipped_logits_mask = resampling_logits > torch.quantile(
                resampling_logits,
                1 - float(self.hparams.sampling_config.clip_reweighting_logits),
            )
            proposal_samples = proposal_samples[~clipped_logits_mask]
            proposal_samples_energy = proposal_samples_energy[~clipped_logits_mask]
            resampling_logits = resampling_logits[~clipped_logits_mask]
            logging.info("Clipped logits for resampling")

        _, resampling_index = resample(proposal_samples, resampling_logits, return_index=True)

        reweighted_data = SamplesData(
            self.datamodule.unnormalize(proposal_samples[resampling_index]),
            proposal_samples_energy[resampling_index],
            logits=resampling_logits,
        )

        if self.hparams.sampling_config.get("load_samples_path", None) is not None:
            load_samples_path_smc = self.hparams.sampling_config.load_samples_path.replace("samples", "smc_samples")
        else:
            load_samples_path_smc = None

        if load_samples_path_smc and os.path.exists(load_samples_path_smc):
            logging.info(f"Loading SMC samples from {load_samples_path_smc}")
            smc_samples_dict = torch.load(load_samples_path_smc, map_location=self.device)
            smc_samples = smc_samples_dict["samples"]
            smc_logits = smc_samples_dict["logits"]
            smc_data = SamplesData(
                self.datamodule.unnormalize(smc_samples),
                energy_fn(smc_samples),
                logits=smc_logits,
            )
        elif self.smc_sampler is not None and self.smc_sampler.enabled:
            logging.info("SMC sampling enabled")

            num_smc_samples = min(self.hparams.sampling_config.num_smc_samples, len(proposal_samples))

            # Generate smc samples and record time
            torch.cuda.synchronize()
            start_time = time.time()

            # TODO: Make conditional proposal energy
            cond_proposal_energy = lambda _x: self.proposal_energy(_x, permutations=permutations, encodings=encodings)
            smc_samples, smc_logits = self.smc_sampler.sample(
                proposal_samples[:num_smc_samples], cond_proposal_energy, energy_fn
            )  # already returned resampled
            torch.cuda.synchronize()
            time_duration = time.time() - start_time
            self.log(f"{prefix}/smc/samples_walltime", time_duration, sync_dist=True)
            self.log(f"{prefix}/smc/samples_per_second", len(smc_samples) / time_duration, sync_dist=True)

            # Save samples to disk
            smc_samples_dict = {
                "smc_samples": smc_samples,
                "smc_logits": smc_logits,
            }
            if self.local_rank == 0:
                if output_dir is None:
                    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                    os.makedirs(f"{output_dir}/{prefix}", exist_ok=True)

                torch.save(smc_samples_dict, f"{output_dir}/{prefix}/smc_samples.pt")
                logging.info(f"Saving {len(smc_samples)} samples to {output_dir}/{prefix}_smc_samples.pt")

            # Datatype for easier metrics and plotting
            smc_data = SamplesData(
                self.datamodule.unnormalize(smc_samples),
                energy_fn(smc_samples),
                logits=smc_logits,
            )
        else:
            smc_data = None

        if self.local_rank == 0:
            # log dataset metrics
            metrics.update(
                self.datamodule.metrics_and_plots(
                    self.log_image,
                    sequence,
                    true_data,
                    proposal_data,
                    reweighted_data,
                    smc_data,
                    tica_model,
                    prefix=prefix,
                )
            )
        else:
            metrics = {}
        return metrics

    def generate_and_resample(self, num_proposal_samples: int = None):
        assert self.datamodule.test_sequences is not None, "Eval sequence name should be set"
        assert len(self.datamodule.test_sequences) == 1, "Can only self-refine on 1 test sequence at a time."
        assert self.datamodule.buffer is not None, "Need to have buffer instantiated in datamodule for self-consumption"

        if num_proposal_samples is None:
            num_proposal_samples = self.hparams.sampling_config.num_proposal_samples

        # on first step, we need to prepare the eval encoding
        _, permutations, eval_encoding, energy_fn, _ = self.datamodule.prepare_eval(self.datamodule.test_sequences[0])

        proposal_samples, proposal_log_p, _ = self.batched_generate_samples(
            num_proposal_samples, permutations=permutations, encodings=eval_encoding, log_invert_error=False
        )

        # Compute energy
        proposal_samples_energy = energy_fn(proposal_samples)

        # Compute proposal center of mass std
        coms = proposal_samples.mean(dim=1, keepdim=False)
        proposal_com_std = coms.std()
        # TODO little scary relying on this class attribute! - gets used in self.proposal_energy
        # when use_com_adjustment=True
        self.proposal_com_std = proposal_com_std

        # Apply CoM adjustment to energy, this must be done here for compatibility with CNFs
        if self.hparams.sampling_config.get("use_com_adjustment", False):
            proposal_log_p = proposal_log_p + self.com_energy_adjustment(proposal_samples)

        # Compute resampling index
        resampling_logits = -proposal_samples_energy - proposal_log_p

        # Filter samples based on logit clipping - this affects both IS and SMC
        if self.hparams.sampling_config.clip_reweighting_logits:
            clipped_logits_mask = resampling_logits > torch.quantile(
                resampling_logits,
                1 - float(self.hparams.sampling_config.clip_reweighting_logits),
            )
            proposal_samples = proposal_samples[~clipped_logits_mask]
            proposal_samples_energy = proposal_samples_energy[~clipped_logits_mask]
            resampling_logits = resampling_logits[~clipped_logits_mask]
            logging.info("Clipped logits for resampling")

        _, resampling_index = resample(proposal_samples, resampling_logits, return_index=True)
        reweighted_samples = self.datamodule.unnormalize(proposal_samples[resampling_index])
        return reweighted_samples

    def on_fit_start(self) -> None:
        """
        Called at the very beginning of the fit loop, after setup() has been called.
        Here we make a copy of the model to serve as a teacher during self-refinement. 
        This ensures the "student" model does not drift to far from the initial model (teacher).
        """

        if self.hparams.use_distill_loss and self.hparams.self_refinement:
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
        if self.hparams.get("self_refinement", False):
            logging.info(
                f"Generating {self.hparams.sampling_config.num_self_refinement_proposal_samples} Samples"
                " for self-consumption"
            )
            self.net.eval()

            if self.hparams.ema_decay > 0:
                self.net.backup()
                self.net.copy_to_model()
                
                with torch.no_grad():
                    samples = self.generate_and_resample(
                        num_proposal_samples=self.hparams.sampling_config.num_self_refinement_proposal_samples,
                    )

                self.net.restore_to_model()

            else:
                 with torch.no_grad():
                    samples = self.generate_and_resample(
                        num_proposal_samples=self.hparams.sampling_config.num_self_refinement_proposal_samples,
                    )

            self.net.train()

            # add the IS samples into the buffer
            self.datamodule.data_train.buffer.add(samples, self.datamodule.test_sequences[0])

            # save the buffer into memory
            self.datamodule.save_buffer()


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
        self.on_eval_epoch_end(self.val_metrics, "val")
        logging.info("Validation epoch end")

    def on_test_epoch_end(self) -> None:
        self.on_eval_epoch_end(self.test_metrics, "test")
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

    # https://github.com/Lightning-AI/pytorch-lightning/issues/1462
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

    def proposal_energy(self, x: torch.Tensor) -> torch.Tensor:
        # x is considered to be a sample from the proposal distribution
        raise NotImplementedError

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("teacher.")}


if __name__ == "__main__":
    _ = TransferableBoltzmannGeneratorLitModule(None, None, None, None)
