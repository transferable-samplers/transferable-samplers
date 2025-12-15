import math

import torch

from src.models.samplers.base_sampler import SMCSampler


class SMCSamplerULA(SMCSampler):
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
        adaptive_step_size: bool = False,
    ):
        super().__init__(
            log_image_fn=log_image_fn,
            batch_size=batch_size,
            langevin_eps=langevin_eps,
            num_timesteps=num_timesteps,
            ess_threshold=ess_threshold,
            warmup=warmup,
            enabled=enabled,
            do_energy_plots=do_energy_plots,
            log_freq=log_freq,
            input_energy_cutoff=input_energy_cutoff,
            systematic_resampling=systematic_resampling,
            adaptive_step_size=adaptive_step_size,
        )

    def mcmc_kernel(self, source_energy, target_energy, t, x, logw, dt):
        # get step size for langevin
        eps = self.langevin_eps_fn(t)

        # get the energy gradients
        energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(
            source_energy, target_energy, t, x, return_t_grad=True
        )
        dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
        dlogw = -energy_grad_t * dt

        x = x + dx
        logw = logw + dlogw

        # acceptance rate is always 1
        return x, logw, torch.ones(1).mean()
