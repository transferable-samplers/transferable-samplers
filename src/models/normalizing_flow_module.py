import math
from typing import Optional

import scipy
import torch

from src.models.transferable_boltzmann_generator_module import TransferableBoltzmannGeneratorLitModule


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

class NormalizingFlowLitModule(BaseLightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: LightningDataModule,
        evaluator: Evaluator,
        ema_decay: float,
        compile: bool,
        sampler: Sampler,
        use_com_adjustment: bool = False,
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`."""
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            datamodule=datamodule,
            evaluator=evaluator,
            ema_decay=ema_decay,
            compile=compile,
        )
        assert not self.hparams.mean_free_prior, "Mean free prior is not supported for normalizing flows"
        self.eval_encodings = None
        self.eval_energy = None

    def model_step(self, batch: torch.Tensor, log: bool = True) -> torch.Tensor:
        x1 = batch["x"]
        encodings = batch.get("encodings")
        permutations = batch.get("permutations")
        mask = batch.get("mask")

        x0, dlogp = self.net(x1, permutations=permutations, encodings=encodings, mask=mask)

        loss = self.prior.energy(x0, mask=mask).mean() - dlogp.mean()

        if log:
            self.log("train/mle_loss", loss.item(), prog_bar=True, sync_dist=True)

        if self.hparams.use_distill_loss and self.hparams.self_improve and self.training:
            with torch.no_grad():
                x0_teacher, dlogp_teacher = self.teacher(x1, permutations=permutations, encodings=encodings, mask=mask)

            prior_log_p = -self.prior.energy(x0_teacher, mask=mask)
            logq_teacher = prior_log_p + dlogp_teacher

            logq = -self.prior.energy(x0, mask=mask) + dlogp
            distill_loss = (logq - logq_teacher).pow(2).mean()
            loss = loss + self.hparams.distill_weight * distill_loss
            if log:
                self.log("finetune/distill_loss", distill_loss.item(), prog_bar=True, sync_dist=True)

        if self.hparams.energy_kl_weight:
            assert self.hparams.eval_sequence is not None, "Eval sequence name should be set"

            if self.eval_encodings is None:
                # on first step, we need to prepare the eval encodings
                model_inputs, _, self.eval_energy = self.datamodule.prepare_eval(self.hparams.eval_sequence)
                self.eval_encodings = model_inputs.encodings

            samples, log_q_theta, _ = self.generate_samples(
                self.hparams.energy_kl_batch_size,
                encodings=self.eval_encodings,
                permutations=self.eval_permutations,
            )

            log_p = -self.eval_energy(samples)

            if log:
                self.log("train/log_p_mean", log_p.mean(), prog_bar=True, sync_dist=True)
                self.log("train/log_p_median", log_p.median(), prog_bar=True, sync_dist=True)

                self.log("train/log_q_theta_mean", log_q_theta.mean(), prog_bar=True, sync_dist=True)
                self.log("train/log_q_theta_median", log_q_theta.median(), prog_bar=True, sync_dist=True)

            num_atoms = self.eval_encodings["atom_type"].size(0)
            data_dim = num_atoms * self.datamodule.hparams.num_dimensions
            energy_loss = (log_q_theta - log_p).mean() / data_dim

            loss = loss + self.hparams.energy_kl_weight * energy_loss

            if log:
                self.log("train/energy_loss", energy_loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def invertibility_metrics(self, z, z_recon):

        metrics = {}

        metrics["mse"] = torch.mean((z - z_recon) ** 2)
        metrics["max_abs"] = torch.max(abs(z - z_recon))
        metrics["mean_abs"] = torch.mean(abs(z - z_recon))
        metrics["median_abs"] = torch.median(abs(z - z_recon))

        for cutoff in [0.01, 0.001, 0.0001]:
            metrics[f"pointwise_fail_at_{cutoff}"] = torch.sum(abs(z - z_recon) > cutoff)
            metrics[f"sample_fail_at_{cutoff}"] = (torch.sum(abs(z - z_recon) > cutoff, dim=1) > 0).sum()

        return metrics

    def sample_proposal(
        self,
        num_samples: int,
        cond_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from the proposal distribution q_theta(x).
    
        Args:
            num_samples: Number of samples to generate.
            cond_inputs: Conditioning inputs passed to the flow.
    
        Returns:
            x: Generated samples in data space.
            log_q_x: Log proposal density log q(x).
            z: Latent prior samples used to generate x.
            invertibility_metrics: Metrics comparing z and z_recon.
        """
    
        encodings = cond_inputs.get("encodings")
        permutations = cond_inputs.get("permutations")
    
        if encodings is None:
            # Handles the non-transferable case
            num_atoms = self.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)
    
        data_dim = num_atoms * self.datamodule.hparams.num_dimensions
        local_batch_size = num_samples // self.trainer.world_size
    
        z = self.prior.sample(local_batch_size, num_atoms, device=self.device)
        log_p_z = -self.prior.energy(z) * data_dim # convert mean → sum
    
        _permutations = self.batchify_model_input(permutations)
        _encodings = self.batchify_model_input(encodings)
    
        x = self.net.reverse(
            z,
            permutations=_permutations,
            encodings=_encodings,
        )
    
        z_recon, dlogp = self.net.forward(
            x,
            permutations=_permutations,
            encodings=_encodings,
        )
        dlogp = dlogp * data_dim  # convert mean → sum
    
        log_q_x = log_p_z.view(-1) + dlogp.view(-1)
    
        invertibility_metrics = self.invertibility_metrics(z, z_recon)
    
        x        = self.all_gather_batch_dim(x)
        log_q_x  = self.all_gather_batch_dim(log_q_x)
        z        = self.all_gather_batch_dim(z)

        return x, log_q_x, z, invertibility_metrics

    def proposal_energy_fn(self, x: torch.Tensor, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor:

        encodings = model_inputs.get("encodings")
        permutations = model_inputs.get("permutations")

        data_dim = x.shape[1] * self.datamodule.hparams.num_dimensions  # is the product num_atoms * num_dimensions
        _encodings = self.batchify_model_input(encodings, x.shape[0]) if encodings is not None else None
        _permutations = self.batchify_model_input(permutations, x.shape[0]) if permutations is not None else None

        z_recon, dlogp = self.net.forward(x, _permutations, encodings=_encodings)

        dlogp = dlogp.view(-1) * data_dim  # rescale from mean to sum
        log_p_z = -self.prior.energy(z_recon).view(-1) * data_dim  # rescale from mean to sum

        log_q_x = log_p_z + dlogp

        proposal_energy = - log_q_x

        return proposal_energy

    def com_energy_adjustment_fn(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        com = x.mean(dim=1, keepdim=False)
        com_norm = com.norm(dim=-1)
        com_energy = com_norm**2 / (2 * sigma**2) - torch.log(
            com_norm**2 / (math.sqrt(2) * sigma**3 * scipy.special.gamma(3 / 2))
        )
        return com_energy
