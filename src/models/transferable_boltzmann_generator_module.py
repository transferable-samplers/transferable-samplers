import inspect
import logging
import os
import statistics as stats
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torchmetrics
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.evaluation.metrics_and_plots import metrics_and_plots
from src.models.base_lightning_module import BaseLightningModule
from src.models.neural_networks.ema import EMA
from src.models.priors import NormalDistribution
from src.models.samplers.base_sampler import SMCSampler
from src.models.utils import get_symmetry_change, resample
from src.utils.data_types import SamplesData

logger = logging.getLogger(__name__)


class TransferableBoltzmannGeneratorLitModule(BaseLightningModule):
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
        output_dir: str = "",
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

        self.smc_sampler = smc_sampler
        if self.smc_sampler is not None:
            self.smc_sampler.log_image_fn = self.log_image

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

        self.output_dir = output_dir