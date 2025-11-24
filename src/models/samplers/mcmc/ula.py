import math
import torch

def ula(e_t_fn, grad_e_t_fn, t, x, logw, dt, eps):
    # get step size for langevin
    dx = -eps * grad_e_t_fn(t, x) + math.sqrt(2 * eps) * torch.randn_like(x)
    dlogw = -grad_e_t_fn(t, x) * dt

    x = x + dx
    logw = logw + dlogw

    return x, logw, torch.ones(1).mean()
