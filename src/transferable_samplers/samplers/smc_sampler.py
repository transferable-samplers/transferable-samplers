# Consider the Euler-Maruyama discretization of the overdamped Langevin SDE:
#
#     X_{k+1} = X_k - h grad_E_t(X_k) + sqrt(2h) ξ,
#     ξ ~ N(0, I),
#
# Where E_t is a time-varying energy interpolation between E_source and E_target.
#
# For continuous-time AIS / SMC, the proposal Langevin step size is coupled with
# the annealing-time increment dt as follows:
#
#     h = eps * dt,
#
# where dt is the annealing-time increment and eps controls the diffusion
# strength per unit annealing time; this ensures convergence to the underlying
# SDE and preservation of intermediate marginals as dt → 0.
#
# In discrete-time MALA, the Metropolis–Hastings correction enforces
# detailed balance, so the proposal scale need not depend on dt.
# However, for consistency across samplers we still use h = eps * dt.
# Consequently, changing num_annealing_steps (and therefore dt) changes
# the effective MALA step size h.
#
# For further details on continuous-time AIS see:
# Proposition 1 of https://arxiv.org/pdf/2410.02711
# Appendix D.2 of https://arxiv.org/pdf/2508.18175
from __future__ import annotations

from typing import Any

import torch

from transferable_samplers.evaluation.metrics.ess import normalized_ess
from transferable_samplers.samplers.base_sampler import BaseSampler
from transferable_samplers.samplers.filtering import filter_by_energy_cutoff, filter_by_logw_quantile
from transferable_samplers.samplers.resampling import resampling_idx
from transferable_samplers.samplers.smc.mcmc import mcmc_kernel
from transferable_samplers.samplers.smc.smc_particles import SMCParticles, all_gather_particles
from transferable_samplers.utils.dataclasses import SamplesData, SourceEnergy, TargetEnergy
from transferable_samplers.utils.dist_utils import (
    all_gather_cat,
    broadcast_tensor,
    get_rank,
    get_world_size,
    shard_tensor,
)
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class SMCSampler(BaseSampler):
    """Sequential Monte Carlo sampler.

    Generates proposals, then refines with SMC using MCMC kernels.
    DDP-aware: particles are sharded across ranks for MCMC,
    gathered globally for ESS checks and resampling.

    Args:
        num_samples: Total number of proposal samples to generate.
        init_eps: Initial diffusion step size (h = eps * dt per annealing step).
        use_metropolis: If True, apply Metropolis-Hastings correction (MALA).
        num_annealing_steps: Number of discrete annealing steps from t=0 to t=1.
        adaptive_step_size: If True, adapt eps based on acceptance rate.
        ess_threshold: Resample when normalized ESS drops below this threshold.
            Set negative to disable intermediate resampling.
        resampling_method: Resampling strategy (``"multinomial"`` or ``"systematic"``).
        energy_cutoff_filter: Drop proposals with target energy above this value.
        logw_quantile_filter: Drop the top fraction of proposals by importance weight.
        log_traj_freq: Save particle snapshots every this many steps for diagnostics.
    """

    def __init__(
        self,
        num_samples: int,
        init_eps: float = 1e-7,
        use_metropolis: bool = False,
        num_annealing_steps: int = 100,
        adaptive_step_size: bool = False,
        ess_threshold: float = -1.0,
        resampling_method: str = "multinomial",
        energy_cutoff_filter: float | None = None,
        logw_quantile_filter: float | None = None,
        log_traj_freq: int = 10,
    ) -> None:
        super().__init__(num_samples)
        logger.warning(
            "init_eps has replaced langevin_eps, and has been reparameterized such that langevin_eps = init_eps * dt"
        )

        self.init_eps = init_eps
        self.use_metropolis = use_metropolis
        self.num_annealing_steps = num_annealing_steps
        self.adaptive_step_size = adaptive_step_size
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method
        self.energy_cutoff_filter = energy_cutoff_filter
        self.logw_quantile_filter = logw_quantile_filter
        self.log_traj_freq = log_traj_freq

    @torch.no_grad()  # sampling path, no training gradients needed
    # pyrefly: ignore [bad-override]
    def sample(
        self,
        source_energy: SourceEnergy,
        target_energy: TargetEnergy,
    ) -> tuple[dict[str, SamplesData], dict[str, Any]]:
        # Generate proposal
        world_size = get_world_size()
        loc_num_samples = self.num_samples // world_size
        loc_samples, loc_E_source = source_energy.sample(loc_num_samples)

        # Compute energy (on each rank, for local samples)
        loc_E_target = target_energy.energy(loc_samples)

        # All gather across ranks
        samples = all_gather_cat(loc_samples)
        E_source = all_gather_cat(loc_E_source)
        E_target = all_gather_cat(loc_E_target)

        # Store for evaluation / plotting
        proposal_data = SamplesData(samples, E_target)

        # Filter by energy cutoff (all ranks do same filtering to maintain same shape)
        if self.energy_cutoff_filter is not None:
            samples, E_source, E_target = filter_by_energy_cutoff(
                samples, E_source, E_target, self.energy_cutoff_filter
            )
            logger.info("Clipping energies")

        # Clip by logit quantile (all ranks do same filtering to maintain same shape)
        if self.logw_quantile_filter is not None:
            samples, E_source, E_target = filter_by_logw_quantile(
                samples, E_source, E_target, self.logw_quantile_filter
            )
            logger.info("Clipped proposal logw for SMC initialisation")

        # Shard across DDP ranks (no-op otherwise)
        n_total = (len(samples) // world_size) * world_size  # trim to be divisible
        lineage = torch.arange(n_total, device=samples.device)  # track original sample indices for diagnostics
        X = shard_tensor(samples[:n_total])
        loc_lineage = shard_tensor(lineage)

        # Initialize SMCParticles with local samples and energies
        loc_E_source, loc_E_source_grad = source_energy.energy_and_grad(X)
        loc_E_target, loc_E_target_grad = target_energy.energy_and_grad(X)
        loc_particles = SMCParticles(
            x=X,
            logw=torch.zeros(X.shape[0], device=X.device),
            lineage=loc_lineage,
            E_source=loc_E_source,
            E_target=loc_E_target,
            E_source_grad=loc_E_source_grad,
            E_target_grad=loc_E_target_grad,
        )

        eps = self.init_eps
        timesteps = torch.linspace(0, 1, self.num_annealing_steps + 1)
        trajectory: list[SMCParticles] = []
        diagnostics: dict[str, list] = {
            "t": [],
            "ess": [],
            "eps": [],
            "acceptance_rate": [],
        }

        # Main loop
        t_previous = 0.0
        for j, t in enumerate(timesteps[:-1]):
            logger.info(f"SMC step {j + 1}/{self.num_annealing_steps}, t={t:.3f}, eps={eps:.2e}")

            dt = t - t_previous
            loc_particles, acceptance_rate = mcmc_kernel(
                loc_particles,
                source_energy,
                target_energy,
                t,
                dt,
                eps,
                use_metropolis=self.use_metropolis,
            )

            if loc_particles.x.isnan().any():
                raise ValueError("X has NaNs")
            elif loc_particles.logw.isnan().any():
                raise ValueError("logw has NaNs")

            if self.adaptive_step_size:
                eps = self._adapt_eps(eps, acceptance_rate)

            ess = normalized_ess(all_gather_cat(loc_particles.logw))

            # Track diagnostics at every step
            diagnostics["t"].append(t)
            diagnostics["ess"].append(ess)
            diagnostics["eps"].append(eps)
            diagnostics["acceptance_rate"].append(acceptance_rate)

            # Collect particle snapshot at log_traj_freq intervals
            if not (j + 1) % self.log_traj_freq:
                temp_particles = all_gather_particles(loc_particles)
                trajectory.append(temp_particles)

            # Resampling when ESS drops below threshold, or always on the final step
            is_final = j + 1 == self.num_annealing_steps
            if ess < self.ess_threshold or is_final:
                particles = all_gather_particles(loc_particles)  # global
                if get_rank() == 0:
                    indices = resampling_idx(particles.logw, self.resampling_method)
                else:
                    indices = torch.zeros(len(particles), dtype=torch.long, device=loc_particles.x.device)
                indices = broadcast_tensor(indices, src=0)
                resampled = particles[indices]
                resampled.logw = torch.zeros(len(resampled), device=resampled.x.device)
                trajectory.append(resampled)  # global
                if not is_final:
                    resampled = shard_tensor(resampled)
                loc_particles = resampled
                logger.info(f"Resampled particles at t={t:.3f} with ESS={ess:.2f}")

            t_previous = t

        unique_ratio = loc_particles.lineage.unique().numel() / len(loc_particles)
        logger.info(f"Fraction of Original Samples: {unique_ratio:.2%}")

        smc_data = SamplesData(loc_particles.x, loc_particles.E_target, logw=loc_particles.logw)
        # pyrefly: ignore [bad-return]
        return {"proposal": proposal_data, "smc": smc_data}, {"trajectory": trajectory, "diagnostics": diagnostics}

    @staticmethod
    def _adapt_eps(eps: float, acceptance_rate: float | torch.Tensor) -> float:
        """Adapt eps based on acceptance rate."""
        if acceptance_rate > 0.6:
            return eps * 1.1
        elif acceptance_rate < 0.55:
            return eps / 1.1
        return eps
