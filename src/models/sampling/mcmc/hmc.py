import torch


def leapfrog(grad_e_t_fn, t, x, v, dt):
    v = v - 0.5 * dt * grad_e_t_fn(t, x)
    x = x + dt * v
    v = v - 0.5 * dt * grad_e_t_fn(t, x)
    return x, v


def hmc(e_t_fn, grad_e_t_fn, t, x, logw, dt, eps):
    norm = lambda _v: torch.sum(_v**2, dim=-1)
    v = torch.randn_like(x)
    s = torch.max(torch.zeros_like(t), t - dt)
    # log w = log w + log p_t(x_{t-1}) - log p_{t-1}(x_{t-1})
    dlogw = -e_t_fn(t, x) + e_t_fn(s, x)

    # update the samples
    x_proposal, v_proposal = leapfrog(grad_e_t_fn, t, x, v, eps)

    # metropolis-hastings
    logp = -0.5 * norm(v_proposal) + 0.5 * norm(v) - e_t_fn(t, x_proposal) + e_t_fn(t, x)
    u = torch.rand_like(logp)
    mask = (logp > torch.log(u))[..., None].float()
    x = mask * x_proposal + (1 - mask) * x

    # update weights
    logw = logw + dlogw
    acceptance_rate = mask.mean()

    return x, logw, acceptance_rate
