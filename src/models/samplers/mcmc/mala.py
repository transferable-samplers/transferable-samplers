import math

import torch


def mala_kernel(energy_interpolation_fn, energy_interpolation_grad_fn, t, x, logw, dt, eps):
    """MALA (Metropolis-Adjusted Langevin Algorithm) MCMC kernel step.

    Args:
        energy_interpolation_fn: (t, x) -> energy
        energy_interpolation_grad_fn: (t, x) -> (x_grad, t_grad)
        t: current SMC time (scalar tensor)
        x: particles (batch, atoms, 3)
        logw: log importance weights (batch,)
        dt: time step size (scalar tensor)
        eps: Langevin step size (scalar)

    Returns:
        (x, logw, acceptance_rate)
    """
    # get the energy gradients
    energy_grad_x, _ = energy_interpolation_grad_fn(t, x)
    dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
    x_proposal = x + dx
    s = torch.max(torch.zeros_like(t), t - dt)

    # log w = log w + log p_t(x_{t-1}) - log p_{t-1}(x_{t-1})
    dlogw = -energy_interpolation_fn(t, x) + energy_interpolation_fn(s, x)

    # metropolis hastings
    energy_grad_x_proposal, _ = energy_interpolation_grad_fn(t, x_proposal)
    E_proposal = energy_interpolation_fn(t, x_proposal)
    E = energy_interpolation_fn(t, x)
    logp = -E_proposal + E
    logp += (
        -0.5
        * torch.sum(((x - x_proposal + eps * energy_grad_x_proposal) ** 2).reshape(x.shape[0], -1), dim=-1)
        / (2 * eps)
    )
    logp -= (
        -0.5
        * torch.sum(((x_proposal - x + eps * energy_grad_x) ** 2).reshape(x_proposal.shape[0], -1), dim=-1)
        / (2 * eps)
    )

    u = torch.rand_like(logp)
    mask = (logp > torch.log(u))[..., None, None].float()
    x = mask * x_proposal + (1 - mask) * x
    logw = logw + dlogw
    acceptance_rate = mask.mean()

    return x, logw, acceptance_rate
