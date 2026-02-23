"""MCMC kernels for SMC annealing: Langevin proposal and Metropolis-Hastings acceptance."""

import torch

from src.models.samplers.smc_utils import SMCParticles, merge_particles
from src.utils.dataclasses import SourceEnergy, TargetEnergy


def langevin_proposal(particles: SMCParticles, source: SourceEnergy, target: TargetEnergy, t, dt, sigma):
    """Langevin proposal step with SMC weight update.

    Uses the standard Langevin SDE:
        dx = -(sigma^2/2) * dt * grad_E + sigma * sqrt(dt) * dW

    For linear interpolation E(t,x) = (1-t)*E_source + t*E_target:
      - Spatial gradient: (1-t)*grad_source + t*grad_target
      - Weight update: dE/dt = E_target - E_source (exact, no autograd needed)

    Computes energies and gradients at the proposed x so the returned
    SMCParticles is ready for a Metropolis-Hastings step.

    Args:
        particles: SMCParticles with positions, weights, energies, and gradients.
        source: SourceEnergy with energy_and_grad callable.
        target: TargetEnergy with energy_and_grad callable.
        t: current SMC time (scalar tensor)
        dt: time step size (scalar tensor)
        sigma: Langevin diffusion coefficient (scalar)

    Returns:
        SMCParticles with proposed x, updated logw, and energies/grads at proposed x.
    """
    energy_grad_x = (1 - t) * particles.E_source_grad + t * particles.E_target_grad
    dx = -(sigma**2 / 2) * dt * energy_grad_x + sigma * torch.sqrt(dt) * torch.randn_like(particles.x)

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


def metropolis_accept(current: SMCParticles, proposal: SMCParticles, t, dt, sigma):
    """Metropolis-Hastings acceptance step for Langevin proposals.

    Computes the interpolated energies and gradients at both x and x_proposal,
    then accepts/rejects based on the MH ratio accounting for the asymmetry
    of the Langevin proposal kernel q(x'|x) vs q(x|x').

    Args:
        current: SMCParticles at current x (with energies/grads).
        proposal: SMCParticles at proposed x (with energies/grads).
        t: current SMC time (scalar tensor)
        sigma: Langevin diffusion coefficient (scalar)
        dt: time step size (scalar tensor)

    Returns:
        (accepted: SMCParticles, acceptance_rate)
    """
    energy_grad_x = (1 - t) * current.E_source_grad + t * current.E_target_grad
    energy_grad_x_proposal = (1 - t) * proposal.E_source_grad + t * proposal.E_target_grad
    E = (1 - t) * current.E_source + t * current.E_target
    E_proposal = (1 - t) * proposal.E_source + t * proposal.E_target

    # TODO discuss the parameter h versus the paper.
    h = (sigma**2 / 2) * dt

    logp = -E_proposal + E
    logp += (
        -0.5
        * torch.sum(((current.x - proposal.x + h * energy_grad_x_proposal) ** 2).reshape(current.x.shape[0], -1), dim=-1)
        / (2 * h)
    )
    logp -= (
        -0.5
        * torch.sum(((proposal.x - current.x + h * energy_grad_x) ** 2).reshape(proposal.x.shape[0], -1), dim=-1)
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
    t, dt, sigma,
    use_metropolis: bool = False,
):
    """Run one MCMC kernel step: Langevin proposal + optional MH acceptance.

    Args:
        particles: SMCParticles with current state (energies/grads already computed).
        source: SourceEnergy with energy_and_grad callable.
        target: TargetEnergy with energy_and_grad callable.
        t: current SMC time (scalar tensor)
        dt: time step size (scalar tensor)
        sigma: Langevin diffusion coefficient (scalar)
        use_metropolis: whether to apply Metropolis-Hastings correction.

    Returns:
        (result: SMCParticles, acceptance_rate)
    """
    proposed = langevin_proposal(particles, source, target, t, dt, sigma)

    if use_metropolis:
        return metropolis_accept(particles, proposed, t, dt, sigma)

    return proposed, torch.ones(1).mean()
