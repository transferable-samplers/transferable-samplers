import torch

from src.models.samplers.base_sampler import SMCSampler


class SMCSamplerHMC(SMCSampler):
    def leapfrog(self, source_energy, target_energy, t, x, v, dt):
        grad_energy = lambda _x: self.linear_energy_interpolation_gradients(source_energy, target_energy, t, _x)
        v = v - 0.5 * dt * grad_energy(x)
        x = x + dt * v
        v = v - 0.5 * dt * grad_energy(x)
        return x, v

    def mcmc_kernel(self, source_energy, target_energy, t, x, logw, dt):
        energy_fn = lambda _t, _x: self.linear_energy_interpolation(source_energy, target_energy, _t, _x)
        norm = lambda _v: torch.sum(_v**2, dim=-1)

        # get step size for langevin
        eps = self.langevin_eps_fn(t)

        # sample momentum from standard gaussian
        v = torch.randn_like(x)
        s = torch.max(torch.zeros_like(t), t - dt)
        # log w = log w + log p_t(x_{t-1}) - log p_{t-1}(x_{t-1})
        dlogw = -energy_fn(t, x) + energy_fn(s, x)

        # update the samples
        x_proposal, v_proposal = self.leapfrog(source_energy, target_energy, t, x, v, eps)

        # metropolis-hastings
        logp = -0.5 * norm(v_proposal) + 0.5 * norm(v) - energy_fn(t, x_proposal) + energy_fn(t, x)
        u = torch.rand_like(logp)
        mask = (logp > torch.log(u))[..., None].float()
        x = mask * x_proposal + (1 - mask) * x

        # update weights
        logw = logw + dlogw
        acceptance_rate = mask.mean()

        return x, logw, acceptance_rate
