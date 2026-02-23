from typing import Optional

import torch

from src.evaluation.metrics.ess import normalized_ess
from src.models.samplers.base_sampler import BaseSampler
from src.models.samplers.utils.mcmc import mcmc_kernel
from src.models.samplers.utils.smc_particles import SMCParticles, all_gather_particles
from src.models.samplers.utils import filter_by_energy_cutoff, filter_by_logit_quantile, resampling_idx
from src.utils.dist_utils import all_gather_cat, broadcast_tensor, get_rank, get_world_size, shard_tensor
from src.utils import pylogger
from src.utils.dataclasses import SamplesData, SourceEnergy, TargetEnergy

logger = pylogger.RankedLogger(__name__, rank_zero_only=False)


class SMCSampler(BaseSampler):
    """Sequential Monte Carlo sampler.

    Generates proposals, then refines with SMC using MCMC kernels.
    DDP-aware: particles are sharded across ranks for MCMC,
    gathered globally for ESS checks and resampling.
    """

    def __init__(
        self,
        num_samples: int,
        init_sigma: float = 1e-7,
        use_metropolis: bool = False,
        num_annealing_steps: int = 100,
        ess_threshold: float = -1.0,
        resampling_method: str = "multinomial",
        adaptive_step_size: bool = False,
        energy_cutoff_filter: Optional[float] = None,
        logit_quantile_filter: Optional[float] = None,
        log_traj_freq: int = 10,
    ):
        super().__init__(num_samples)

        self.energy_cutoff_filter = energy_cutoff_filter
        self.logit_quantile_filter = logit_quantile_filter
        self.init_sigma = init_sigma
        self.use_metropolis = use_metropolis
        self.num_annealing_steps = num_annealing_steps
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method
        self.adaptive_step_size = adaptive_step_size
        self.log_traj_freq = log_traj_freq

    @torch.no_grad()
    def sample(
        self,
        source_energy: SourceEnergy,
        target_energy: TargetEnergy,
    ):
        # ── Generate proposals ────────────────────────────────────────────
        samples, log_q = source_energy.sample(self.num_samples)

        # Compute energies
        target_E = target_energy.energy(samples)

        # All gather
        samples = all_gather_cat(samples)
        log_q = all_gather_cat(log_q)
        target_E = all_gather_cat(target_E)

        # Store for evaluation / plotting
        proposal_data = SamplesData(samples, target_E)

        # ── Filter by energy cutoff ───────────────────────────────────────
        if self.energy_cutoff_filter is not None:
            samples, log_q, target_E = filter_by_energy_cutoff(samples, log_q, target_E, self.energy_cutoff_filter)
            logger.info("Clipping energies")

        # ── Clip by logit quantile ────────────────────────────────────────
        if self.logit_quantile_filter is not None:
            samples, log_q, target_E = filter_by_logit_quantile(samples, log_q, target_E, self.logit_quantile_filter)
            logger.info("Clipped proposal logits for SMC initialisation")

        # ── Shard across DDP ranks (no-op if single GPU) ─────────────────
        world_size = get_world_size()
        n_total = (len(samples) // world_size) * world_size
        lineage = torch.arange(n_total, device=samples.device)
        X = shard_tensor(samples[:n_total])
        local_lineage = shard_tensor(lineage)

        # ── Initialize particles ──────────────────────────────────────────
        E_source, E_source_grad = source_energy.energy_and_grad(X)
        E_target, E_target_grad = target_energy.energy_and_grad(X)
        particles = SMCParticles(
            x=X,
            logw=torch.zeros(X.shape[0], device=X.device),
            lineage=local_lineage,
            E_source=E_source, E_target=E_target,
            E_source_grad=E_source_grad, E_target_grad=E_target_grad,
        )

        sigma = self.init_sigma
        timesteps = torch.linspace(0, 1, self.num_annealing_steps + 1)
        trajectory: list[SMCParticles] = []
        diagnostics: dict[str, list] = {
            "t": [], "ess": [], "sigma": [], "acceptance_rate": [],
        }

        # ── Annealing loop ────────────────────────────────────────────────
        t_previous = 0.0
        for j, t in enumerate(timesteps[:-1]):

            logger.info(f"SMC step {j+1}/{self.num_annealing_steps}, t={t:.3f}, sigma={sigma:.2e}")

            dt = t - t_previous
            particles, acceptance_rate = mcmc_kernel(
                particles, source_energy, target_energy, t, dt, sigma,
                use_metropolis=self.use_metropolis,
            )

            if particles.x.isnan().any():
                raise ValueError("X has NaNs")
            elif particles.logw.isnan().any():
                raise ValueError("logw has NaNs")

            if self.adaptive_step_size:
                sigma = self._adapt_sigma(sigma, acceptance_rate)

            ess = normalized_ess(all_gather_cat(particles.logw))

            # Track diagnostics at every step
            diagnostics["t"].append(t)
            diagnostics["ess"].append(ess)
            diagnostics["sigma"].append(sigma)
            diagnostics["acceptance_rate"].append(acceptance_rate)

            # Collect particle snapshot at log_traj_freq intervals
            if not (j + 1) % self.log_traj_freq:
                trajectory.append(particles)

            # Resampling when ESS drops below threshold, or always on the final step
            is_final = j + 1 == self.num_annealing_steps
            if ess < self.ess_threshold or is_final:
                global_particles = all_gather_particles(particles)
                if get_rank() == 0:
                    indices = resampling_idx(global_particles.logw, self.resampling_method)
                else:
                    indices = torch.zeros(len(global_particles), dtype=torch.long, device=particles.x.device)
                indices = broadcast_tensor(indices, src=0)
                resampled = global_particles[indices]
                resampled.logw = torch.zeros(len(resampled), device=resampled.x.device)
                if not is_final:
                    resampled = shard_tensor(resampled)
                particles = resampled
                trajectory.append(particles)
                logger.info(f"Resampled particles at t={t:.3f} with ESS={ess:.2f} and acceptance_rate={acceptance_rate:.2%}")

            t_previous = t

        unique_ratio = particles.lineage.unique().numel() / len(particles)
        logger.info(f"Fraction of Original Samples: {unique_ratio:.2%}")

        smc_data = SamplesData(particles.x, particles.E_target, logits=particles.logw)
        return {"proposal": proposal_data, "smc": smc_data}, {"trajectory": trajectory, "diagnostics": diagnostics}

    # ── Step size scheduling ─────────────────────────────────────────────

    @staticmethod
    def _adapt_sigma(sigma, acceptance_rate):
        """Adapt sigma based on acceptance rate."""
        if acceptance_rate > 0.6:
            return sigma * 1.1
        elif acceptance_rate < 0.55:
            return sigma / 1.1
        return sigma

