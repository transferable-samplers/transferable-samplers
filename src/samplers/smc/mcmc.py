
import torch

from src.sampling.smc import SMCParticles, merge_particles
from src.utils.dataclasses import SourceEnergy, TargetEnergy


def linear_interpolation(source: torch.Tensor, target: torch.Tensor, t) -> torch.Tensor:
    """(1-t)*source + t*target."""
    return (1 - t) * source + t * target


def langevin_proposal(particles: SMCParticles, source: SourceEnergy, target: TargetEnergy, t, dt, eps):
    """Langevin proposal + SMC weight update. Returns particles with energies/grads at x'."""
    h = eps * dt
    E_interp_grad = linear_interpolation(particles.E_source_grad, particles.E_target_grad, t)
    dx = -h * E_interp_grad + torch.sqrt(2 * h) * torch.randn_like(particles.x)

    delta = t - torch.max(torch.zeros_like(t), t - dt)
    dlogw = -(particles.E_target - particles.E_source) * delta

    x_proposal = particles.x + dx
    logw_proposal = particles.logw + dlogw
    E_source_prop, E_source_grad_prop = source.energy_and_grad(x_proposal)
    E_target_prop, E_target_grad_prop = target.energy_and_grad(x_proposal)

    return SMCParticles(
        x=x_proposal,
        logw=logw_proposal,
        lineage=particles.lineage,
        E_source=E_source_prop,
        E_target=E_target_prop,
        E_source_grad=E_source_grad_prop,
        E_target_grad=E_target_grad_prop,
    )


def metropolis_accept(current: SMCParticles, proposal: SMCParticles, t, dt, eps):
    """MH accept/reject for a Langevin proposal, correcting for kernel asymmetry q(x'|x) vs q(x|x').

    The log-acceptance ratio is:

        log α = -E_t(x') + E_t(x)
              + log q(x|x') - log q(x'|x)

    where q is the Langevin kernel N(x - h ∇E_t(x), 2hI) with h = eps * dt.
    Weight update (logw) is set regardless of acceptance (it tracks the annealing schedule, not x).
    """
    E_interp = linear_interpolation(current.E_source, current.E_target, t)
    E_interp_proposal = linear_interpolation(proposal.E_source, proposal.E_target, t)
    E_interp_grad = linear_interpolation(current.E_source_grad, current.E_target_grad, t)
    E_interp_grad_proposal = linear_interpolation(proposal.E_source_grad, proposal.E_target_grad, t)

    h = eps * dt

    logp = -E_interp_proposal + E_interp
    logp += (
        -0.5
        * torch.sum(((current.x - proposal.x + h * E_interp_grad_proposal) ** 2).reshape(current.x.shape[0], -1), dim=-1)
        / (2 * h)
    )
    logp -= (
        -0.5
        * torch.sum(((proposal.x - current.x + h * E_interp_grad) ** 2).reshape(proposal.x.shape[0], -1), dim=-1)
        / (2 * h)
    )

    u = torch.rand_like(logp)
    accept = logp > torch.log(u)

    accepted_proposal = proposal[accept]
    kept_original = current[~accept]
    result = merge_particles(accept, accepted_proposal, kept_original)
    result.logw = proposal.logw

    return result, accept.float().mean()


def mcmc_kernel(
    particles: SMCParticles,
    source: SourceEnergy,
    target: TargetEnergy,
    t, dt, eps,
    use_metropolis: bool = False,
):
    """One MCMC step: Langevin proposal, optionally followed by MH correction."""
    proposed = langevin_proposal(particles, source, target, t, dt, eps)

    if use_metropolis:
        return metropolis_accept(particles, proposed, t, dt, eps)

    return proposed, torch.ones(1).mean()
