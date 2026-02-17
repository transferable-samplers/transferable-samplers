import logging
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from tqdm import tqdm

from src.data.normalization import unnormalize
from src.evaluation.metrics.ess import sampling_efficiency
from src.models.samplers.base_sampler_class import BaseSampler
from src.models.samplers.mcmc import mala_kernel, ula_kernel
from src.utils.dataclasses import ProposalCond, SamplesData
from src.utils.resampling import (
    com_energy_adjustment,
    resample_multinomial,
    resample_systematic,
)

if TYPE_CHECKING:
    from lightning import LightningModule

KERNEL_FNS = {"ula": ula_kernel, "mala": mala_kernel}

logger = logging.getLogger(__name__)


class SMCSampler(BaseSampler):
    """Sequential Monte Carlo sampler.

    Generates proposal via SNIS, then refines with SMC using MCMC kernels.
    DDP-aware: particles are sharded across ranks for MCMC,
    gathered globally for ESS checks and resampling.
    """

    def __init__(
        self,
        num_samples: int,
        proposal_batch_size: int,
        kernel_type: str = "ula",
        langevin_eps: float = 1e-7,
        num_timesteps: int = 100,
        ess_threshold: float = -1.0,
        systematic_resampling: bool = False,
        adaptive_step_size: bool = False,
        warmup: float = 0.0,
        gradient_batch_size: int = 128,
        input_energy_filter_cutoff: Optional[float] = None,
        # SNIS params for initial proposal
        use_com_adjustment: bool = False,
        logit_clip_filter: Optional[float] = None,
        # Plotting
        log_image_fn: Optional[Callable] = None,
        do_energy_plots: bool = False,
        log_freq: int = 10,
    ):
        super().__init__(num_samples, proposal_batch_size)

        if kernel_type not in KERNEL_FNS:
            raise ValueError(f"Unknown kernel_type '{kernel_type}', expected one of {list(KERNEL_FNS)}")

        self.kernel_type = kernel_type
        self.kernel_fn = KERNEL_FNS[kernel_type]
        self.langevin_eps = langevin_eps
        self.num_timesteps = num_timesteps
        self.ess_threshold = ess_threshold
        self.systematic_resampling = systematic_resampling
        self.adaptive_step_size = adaptive_step_size
        self.warmup = warmup
        self.gradient_batch_size = gradient_batch_size
        self.input_energy_filter_cutoff = input_energy_filter_cutoff
        self.use_com_adjustment = use_com_adjustment
        self.logit_clip_filter = logit_clip_filter
        self.log_image_fn = log_image_fn
        self.do_energy_plots = do_energy_plots
        self.log_freq = log_freq

    def sample(
        self,
        model: "LightningModule",
        proposal_cond: Optional[ProposalCond],
        target_energy_fn,
        prefix: str = "",
    ) -> dict[str, SamplesData]:
        # 1. Generate proposal samples (normalized)
        samples, log_q = self.sample_proposal_in_batches(model, self.num_samples, proposal_cond)
        target_energy = target_energy_fn(samples)

        std = model.trainer.datamodule.std
        proposal_data = SamplesData(unnormalize(samples, std), target_energy)

        # 2. SNIS resampling
        adjusted_log_q = log_q
        if self.use_com_adjustment:
            coms = samples.mean(dim=1)
            com_std = coms.std()
            logger.info(f"Applying CoM energy adjustment (com_std={com_std:.4f})")
            adjusted_log_q = log_q + com_energy_adjustment(samples, com_std)

        logits = -target_energy - adjusted_log_q

        if self.logit_clip_filter:
            clipped_mask = logits > torch.quantile(logits, 1 - self.logit_clip_filter)
            samples = samples[~clipped_mask]
            target_energy = target_energy[~clipped_mask]
            logits = logits[~clipped_mask]
            logger.info("Clipped logits for resampling")

        _, resampling_index = resample_multinomial(samples, logits)
        resampled_data = SamplesData(
            unnormalize(samples[resampling_index], std),
            target_energy[resampling_index],
            logits=logits,
        )

        # 3. Build source energy function (with COM adjustment if enabled)
        if self.use_com_adjustment:
            source_energy_fn = lambda x: model.proposal_energy(x, proposal_cond) - com_energy_adjustment(x, com_std)
        else:
            source_energy_fn = lambda x: model.proposal_energy(x, proposal_cond)

        # 4. Run SMC loop on the (possibly clipped) proposal samples
        smc_samples, smc_logits = self._smc_loop(
            samples, source_energy_fn, target_energy_fn, model
        )

        smc_energy = target_energy_fn(smc_samples)
        smc_data = SamplesData(unnormalize(smc_samples, std), smc_energy, logits=smc_logits)

        return {"proposal": proposal_data, "resampled": resampled_data, "smc": smc_data}

    # ── Energy interpolation ─────────────────────────────────────────────

    def linear_energy_interpolation(self, source_energy_fn, target_energy_fn, t, x):
        E_source = source_energy_fn(x)
        E_target = target_energy_fn(x)
        assert E_source.shape == (x.shape[0],), f"Source energy should be flat, got {E_source.shape}"
        assert E_target.shape == (x.shape[0],), f"Target energy should be flat, got {E_target.shape}"
        return (1 - t) * E_source + t * E_target

    def linear_energy_interpolation_gradients(self, source_energy_fn, target_energy_fn, t, x):
        t = t.repeat(x.shape[0]).to(x)

        with torch.set_grad_enabled(True):
            x.requires_grad = True
            t.requires_grad = True
            et = self.linear_energy_interpolation(source_energy_fn, target_energy_fn, t, x)
            t_grad, x_grad = torch.autograd.grad(et.sum(), (t, x))

            assert x_grad.shape == x.shape, "x_grad should have the same shape as x"
            assert t_grad.shape == t.shape, "t_grad should have the same shape as t"

        assert x_grad is not None, "x_grad should not be None"
        assert t_grad is not None, "t_grad should not be None"

        return x_grad.detach(), t_grad.detach()

    # ── Step size scheduling ─────────────────────────────────────────────

    def langevin_eps_fn(self, t):
        if t < self.warmup:
            return (self.langevin_eps * t) / self.warmup
        return self.langevin_eps

    def update_step_size(self, acceptance_rate):
        if acceptance_rate > 0.6:
            self.langevin_eps *= 1.1
        elif acceptance_rate < 0.55:
            self.langevin_eps /= 1.1

    # ── MCMC kernel dispatch ─────────────────────────────────────────────

    def mcmc_kernel(self, source_energy_fn, target_energy_fn, t, x, logw, dt):
        eps = self.langevin_eps_fn(t)
        energy_fn = lambda _t, _x: self.linear_energy_interpolation(source_energy_fn, target_energy_fn, _t, _x)
        grad_fn = lambda _t, _x: self.linear_energy_interpolation_gradients(source_energy_fn, target_energy_fn, _t, _x)
        return self.kernel_fn(energy_fn, grad_fn, t, x, logw, dt, eps)

    # ── Resampling ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _resample(self, x, logw):
        if self.systematic_resampling:
            return resample_systematic(x, logw)
        return resample_multinomial(x, logw)

    # ── Main SMC loop ────────────────────────────────────────────────────

    @torch.no_grad()
    def _smc_loop(self, proposal_samples, source_energy_fn, target_energy_fn, model):
        """Run SMC loop. In DDP mode, particles are sharded across ranks."""
        world_size = model.trainer.world_size if model.trainer else 1

        # Filter by energy cutoff
        if self.input_energy_filter_cutoff is not None:
            energies = target_energy_fn(proposal_samples)
            proposal_samples = proposal_samples[energies < self.input_energy_filter_cutoff]
            logger.info("Clipping energies")

        num_timesteps = self.num_timesteps
        timesteps = torch.linspace(0, 1, num_timesteps + 1)

        # Shard particles across DDP ranks
        if world_size > 1:
            rank = model.local_rank
            chunk_size = len(proposal_samples) // world_size
            X = proposal_samples[rank * chunk_size : (rank + 1) * chunk_size]
        else:
            X = proposal_samples

        A = torch.zeros(X.shape[0], device=X.device)

        # Tracking lists
        A_list = [A]
        ESS_list = [1.0]
        t_list = [timesteps[0]]
        eps_list = [self.langevin_eps_fn(0.0)]
        acceptance_rate_list = [torch.tensor(1.0)]
        survived_lineages = [torch.tensor(1.0)]
        particle_ids = torch.arange(X.shape[0])

        if self.do_energy_plots:
            X_batches = [X[i : i + self.gradient_batch_size] for i in range(0, X.shape[0], self.gradient_batch_size)]
            target_energy_list = [
                np.concatenate([target_energy_fn(xb).cpu().numpy() for xb in X_batches])
            ]
            interpolation_energy_list = [
                np.concatenate([
                    self.linear_energy_interpolation(source_energy_fn, target_energy_fn, timesteps[0], xb).cpu().numpy()
                    for xb in X_batches
                ])
            ]

        t_previous = 0.0
        for j, t in tqdm(enumerate(timesteps[:-1]), total=num_timesteps):
            logger.info(f"Outer loop iteration {j}")

            # Split into gradient-computation batches
            X_batches = [X[i : i + self.gradient_batch_size] for i in range(0, X.shape[0], self.gradient_batch_size)]
            A_batches = [A[i : i + self.gradient_batch_size] for i in range(0, A.shape[0], self.gradient_batch_size)]

            target_energy_batches = []
            interpolation_energy_batches = []
            batch_acceptance_rate_list = []

            dt = t - t_previous
            for batch_idx, (X_batch, A_batch) in enumerate(zip(X_batches, A_batches)):
                if X_batch.isnan().any():
                    raise ValueError("X contains NaNs")

                X_batch, A_batch, acceptance_rate = self.mcmc_kernel(
                    source_energy_fn, target_energy_fn, t, X_batch, A_batch, dt
                )

                X_batches[batch_idx] = X_batch
                A_batches[batch_idx] = A_batch

                if self.do_energy_plots:
                    target_energy_batches.append(target_energy_fn(X_batch).cpu())
                    interpolation_energy_batches.append(
                        self.linear_energy_interpolation(source_energy_fn, target_energy_fn, t, X_batch).cpu()
                    )

                batch_acceptance_rate_list.append(acceptance_rate.view(-1))

            # Concatenate batches
            X = torch.cat(X_batches, dim=0)
            A = torch.cat(A_batches, dim=0)
            acceptance_rate = torch.cat(batch_acceptance_rate_list, dim=0).mean()

            if self.adaptive_step_size:
                self.update_step_size(acceptance_rate)

            assert A.dim() == 1, "A should be a flat vector"

            # Plot on NaN, at log_freq intervals, or at final step
            if X.isnan().any() or A.isnan().any() or not (j + 1) % self.log_freq or j + 1 == num_timesteps:
                if self.log_image_fn is not None:
                    if self.do_energy_plots:
                        self._plot_stepwise_energy(target_energy_list, interpolation_energy_list, t_list)
                        self._plot_stepwise_energy_hist(target_energy_list, interpolation_energy_list, t_list)
                    self._plot_weights(A_list, ESS_list, t_list)
                    self._plot_eps(eps_list, t_list)
                    self._plot_acceptance_rate(acceptance_rate_list, t_list)
                    self._plot_particle_survival(survived_lineages, t_list)

            if X.isnan().any():
                raise ValueError("X has NaNs")
            elif A.isnan().any():
                raise ValueError("A has NaNs")

            # Track metrics
            A_list.append(A)

            # ESS — use global weights in DDP
            if world_size > 1:
                global_A = model.all_gather(A).reshape(-1)
            else:
                global_A = A

            ESS = sampling_efficiency(global_A)
            ESS_list.append(ESS.cpu())
            acceptance_rate_list.append(acceptance_rate.cpu())
            unique_ratio = particle_ids.unique().numel() / len(particle_ids)
            survived_lineages.append(unique_ratio)

            t_list.append(t)
            eps = self.langevin_eps_fn(t)
            eps_list.append(eps)

            if self.do_energy_plots:
                target_energy_list.append(np.concatenate(target_energy_batches))
                interpolation_energy_list.append(np.concatenate(interpolation_energy_batches))

            # Resampling when ESS drops below threshold
            if ESS < self.ess_threshold and not j + 1 == num_timesteps:
                if world_size > 1:
                    global_X = model.all_gather(X).reshape(-1, *X.shape[1:])
                    global_X, indexes = self._resample(global_X, global_A)
                    X = global_X[rank * chunk_size : (rank + 1) * chunk_size]
                else:
                    X, indexes = self._resample(X, A)
                    particle_ids = particle_ids[indexes.cpu()]

                A = torch.zeros(X.shape[0], device=X.device)
                logger.info(f"resampling @ step {j}")

                A_list.append(A)
                if world_size > 1:
                    global_A = model.all_gather(A).reshape(-1)
                else:
                    global_A = A
                ESS = sampling_efficiency(global_A)
                ESS_list.append(ESS.cpu())

                t_list.append(t + 1e-9)
                eps_list.append(eps)
                acceptance_rate_list.append(acceptance_rate.cpu())
                unique_ratio = particle_ids.unique().numel() / len(particle_ids)
                survived_lineages.append(unique_ratio)

                if self.do_energy_plots:
                    X_batches = [
                        X[i : i + self.gradient_batch_size]
                        for i in range(0, X.shape[0], self.gradient_batch_size)
                    ]
                    target_energy_list.append(
                        np.concatenate([target_energy_fn(xb).cpu().numpy() for xb in X_batches])
                    )
                    interpolation_energy_list.append(
                        np.concatenate([
                            self.linear_energy_interpolation(
                                source_energy_fn, target_energy_fn, t + 1e-9, xb
                            ).cpu().numpy()
                            for xb in X_batches
                        ])
                    )

            t_previous = t

        # Final gather + resample
        if world_size > 1:
            X = model.all_gather(X).reshape(-1, *X.shape[1:])
            A = model.all_gather(A).reshape(-1)

        X, indexes = self._resample(X, A)
        particle_ids = particle_ids[indexes.cpu()]
        unique_ratio = particle_ids.unique().numel() / len(particle_ids)
        logger.info(f"resampling @ step {j}")
        logger.info(f"Fraction of Original Samples: {unique_ratio} %")

        smc_samples = X
        smc_logits = A
        assert smc_logits.dim() == 1, "smc_weights should be a flat vector"
        return smc_samples, smc_logits

    # ── Plotting methods ─────────────────────────────────────────────────

    def _plot_stepwise_energy(self, target_energy_list, interpolation_energy_list, t_list):
        stepwise_target_energy_np = np.stack(target_energy_list)
        stepwise_interpolation_energy_np = np.stack(interpolation_energy_list)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        for k in range(stepwise_target_energy_np.shape[1]):
            axs[0].plot(t_list, stepwise_target_energy_np[:, k], linewidth=1, alpha=0.5)
            axs[1].plot(t_list, stepwise_interpolation_energy_np[:, k], linewidth=1, alpha=0.5)

        axs[0].set_xlabel("Time", fontsize=12)
        axs[0].set_ylabel("Target energy", fontsize=12)
        axs[1].set_xlabel("Time", fontsize=12)
        axs[1].set_ylabel("Interpolation energy", fontsize=12)

        plt.tight_layout()
        self.log_image_fn(fig, "langevin/energies")
        plt.close()

    def _plot_stepwise_energy_hist(self, target_energy_list, interpolation_energy_list, t_list):
        stepwise_target_energy_np = np.stack(target_energy_list)
        stepwise_interpolation_energy_np = np.stack(interpolation_energy_list)
        t_np = np.array(t_list)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        data = stepwise_target_energy_np
        bins = np.linspace(data.min(), data.max(), 100)
        histograms = np.array([np.histogram(row, bins=bins)[0] for row in data])
        histograms_normalized = histograms / histograms.sum(axis=1, keepdims=True)
        extent = [t_np.min(), t_np.max(), bins[0], bins[-1]]
        im = axs[0].imshow(
            histograms_normalized.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            norm=LogNorm(
                vmin=histograms_normalized[histograms_normalized > 0].min(),
                vmax=histograms_normalized.max(),
            ),
            cmap="inferno",
        )
        axs[0].set_xlabel("Time", fontsize=12)
        axs[0].set_ylabel("Target energy", fontsize=12)
        fig.colorbar(im, ax=axs[0], label="Log Marginal Density")

        data = stepwise_interpolation_energy_np
        bins = np.linspace(data.min(), data.max(), 100)
        histograms = np.array([np.histogram(row, bins=bins)[0] for row in data])
        histograms_normalized = histograms / histograms.sum(axis=1, keepdims=True)
        extent = [t_np.min(), t_np.max(), bins[0], bins[-1]]
        im = axs[1].imshow(
            histograms_normalized.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            norm=LogNorm(
                vmin=histograms_normalized[histograms_normalized > 0].min(),
                vmax=histograms_normalized.max(),
            ),
            cmap="inferno",
        )
        axs[1].set_xlabel("Time", fontsize=12)
        axs[1].set_ylabel("Interpolation energy", fontsize=12)
        fig.colorbar(im, ax=axs[1], label="Log Marginal Density")

        plt.tight_layout()
        self.log_image_fn(fig, "langevin/energy_histograms")
        plt.close()

    def _plot_weights(self, A_list, ESS_list, t_list):
        A_np = torch.stack(A_list).cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        for k in range(A_np.shape[1]):
            axs[0].plot(t_list, A_np[:, k], linewidth=1, alpha=0.5)
        axs[0].set_xlabel("Time", fontsize=12)
        axs[0].set_ylabel("A", fontsize=12)

        axs[1].plot(t_list, ESS_list, linewidth=1, alpha=0.3)
        axs[1].set_xlabel("Time", fontsize=12)
        axs[1].set_ylabel("ESS", fontsize=12)
        axs[1].set_yscale("log")

        plt.tight_layout()
        self.log_image_fn(fig, "langevin/weights")
        plt.close()

    def _plot_eps(self, eps_list, t_list):
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(t_list, eps_list, linewidth=1, alpha=0.5)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Eps", fontsize=12)
        plt.tight_layout()
        self.log_image_fn(fig, "langevin/eps")
        plt.close()

    def _plot_acceptance_rate(self, acceptance_rate_list, t_list):
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(t_list, acceptance_rate_list, linewidth=1, alpha=0.5)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Acceptance Rate", fontsize=12)
        plt.tight_layout()
        self.log_image_fn(fig, "langevin/acceptance-rate")
        plt.close()

    def _plot_particle_survival(self, survived_lineages, t_list):
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(t_list, survived_lineages, linewidth=1, alpha=0.5)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Survived Lineages (%)", fontsize=12)
        plt.tight_layout()
        self.log_image_fn(fig, "langevin/linage-survived")
        plt.close()
