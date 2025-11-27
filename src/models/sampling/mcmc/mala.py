import math

import torch


def mala(e_t_fn, grad_e_t_fn, t, x, logw, dt, eps):
    energy_grad_x = grad_e_t_fn(t, x)
    dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
    x_proposal = x + dx
    s = torch.max(torch.zeros_like(t), t - dt)

    # log w = log w + log p_t(x_{t-1}) - log p_{t-1}(x_{t-1})
    dlogw = -e_t_fn(t, x) + e_t_fn(s, x)

    # metropolis hastings
    energy_grad_x_proposal = grad_e_t_fn(t, x_proposal)
    E_proposal = e_t_fn(t, x_proposal)
    E = e_t_fn(t, x)
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
