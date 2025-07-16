import math

import torch

from src.models.samplers.base_sampler import SMCSampler


class ContinuousTimeSMCSampler(SMCSampler):
    def __init__(
        self,
        log_image_fn: callable = None,
        batch_size: int = 128,
        langevin_eps: float = 1e-7,
        num_timesteps: int = 100,
        ess_threshold: float = -1.0,
        warmup: float = 0.0,
        enabled: bool = False,
        do_energy_plots: bool = False,
        log_freq: int = 10,
        input_energy_cutoff: float = None,
        systematic_resampling: bool = False,
    ):
        super().__init__()

        self.log_image_fn = log_image_fn
        self.batch_size = batch_size
        self.langevin_eps = langevin_eps
        self.num_timesteps = num_timesteps
        self.ess_threshold = ess_threshold
        self.warmup = warmup
        self.enabled = enabled
        self.do_energy_plots = do_energy_plots
        self.log_freq = log_freq
        self.input_energy_cutoff = input_energy_cutoff
        self.systematic_resampling = systematic_resampling

    def mcmc_kernel(self, source_energy, target_energy, t, x, logw, dt):
        # get step size for langevin
        eps = self.langevin_eps_fn(t)

        # get the energy gradients
        energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(source_energy, target_energy, t, x)
        dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
        dlogw = -energy_grad_t * dt

        x = x + dx
        logw = logw + dlogw

        # acceptance rate is always 1
        return x, logw, torch.ones(1).mean()

    def linear_energy_interpolation(self, source_energy, target_energy, t, x):
        E_source = source_energy(x)
        E_target = target_energy(x)

        assert E_source.shape == (x.shape[0],), f"Source energy should be a flat vector not {E_source.shape}"
        assert E_target.shape == (x.shape[0],), f"Target energy should be a flat vector, not {E_target.shape}"
        energy = (1 - t) * E_source + t * E_target
        return energy

    def linear_energy_interpolation_gradients(self, source_energy, target_energy, t, x):
        t = t.repeat(x.shape[0]).to(x)

        with torch.set_grad_enabled(True):
            x.requires_grad = True
            t.requires_grad = True

            et = self.linear_energy_interpolation(source_energy, target_energy, t, x)

            # assert et.requires_grad, "et should require grad - check the energy function for no_grad"

            # this is a bit hacky but is fine as long as
            # the energy function is defined properly and
            # doesn't mix batch items
            t_grad, x_grad = torch.autograd.grad(et.sum(), (t, x))

            assert x_grad.shape == x.shape, "x_grad should have the same shape as x"
            assert t_grad.shape == t.shape, "t_grad should have the same shape as t"

        assert x_grad is not None, "x_grad should not be None"
        assert t_grad is not None, "t_grad should not be None"

        return x_grad.detach(), t_grad.detach()
