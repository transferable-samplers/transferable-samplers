import math

import torch


def ula_kernel(energy_interpolation_fn, energy_interpolation_grad_fn, t, x, logw, dt, eps):
    """ULA (Unadjusted Langevin Algorithm) MCMC kernel step.

    Args:
        energy_interpolation_fn: (t, x) -> energy (unused by ULA, included for uniform signature)
        energy_interpolation_grad_fn: (t, x) -> (x_grad, t_grad)
        t: current SMC time (scalar tensor)
        x: particles (batch, atoms, 3)
        logw: log importance weights (batch,)
        dt: time step size (scalar tensor)
        eps: Langevin step size (scalar)

    Returns:
        (x, logw, acceptance_rate)
    """
    energy_grad_x, energy_grad_t = energy_interpolation_grad_fn(t, x)
    dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
    dlogw = -energy_grad_t * dt

    x = x + dx
    logw = logw + dlogw

    # acceptance rate is always 1 (no MH step)
    return x, logw, torch.ones(1).mean()
