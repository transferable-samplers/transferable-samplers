import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import scipy.special
import torch


@dataclass(frozen=True)
class SourceEnergyConfig:
    """Configuration for SourceEnergy batch sizes. Passed via Hydra configs."""
    sample_batch_size: int
    energy_batch_size: int
    grad_batch_size: int
    use_com_adjustment: bool = False


@dataclass
class TargetEnergy:
    """Target energy function, provided by the datamodule."""
    energy_fn: Callable  # (x) -> energy (batch,)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_fn(x)

    def energy_and_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            e = self.energy_fn(x)
            g = torch.autograd.grad(e.sum(), x)[0]
        return e.detach(), g.detach()


@dataclass
class SourceEnergy:
    """Source (proposal) energy and sampling, provided by the lightning module.

    Created via BaseLightningModule.build_source_energy(). Handles batching
    internally for sample, energy, and energy_and_grad.
    """
    sample_fn: Callable        # (num_samples) -> (samples, log_q)
    energy_fn: Callable        # (x) -> energy (batch,)
    sample_batch_size: int
    energy_batch_size: int
    grad_batch_size: int
    use_com_adjustment: bool = False

    @staticmethod
    def com_energy_adjustment(x: torch.Tensor) -> torch.Tensor:
        """CoM energy adjustment with std = 1/sqrt(num_atoms).

        Introduced in Prop. 1 of https://arxiv.org/pdf/2502.18462.
        x: (batch, num_atoms, 3) -> adjustment (batch,) to add to log_q / energy.

        NOTE: need to benchmark / implement the fixed version from https://arxiv.org/pdf/2602.03729
        """

        # The mean of N i.i.d. standard Gaussian samples has std = 1/sqrt(N)
        # This is the same as used in the data augmentation.
        com_std_analytic = 1.0 / math.sqrt(x.shape[1])

        # Convert to the equivalent std of the norm of a 3D Gaussian.
        com_norm_std_analytic = com_std_analytic * math.sqrt(3 - 8/math.pi)

        # Compute the Norms of the CoM for each sample in the batch.
        com_norms = x.mean(dim=1).norm(dim=-1)
        return (
            com_norms ** 2 / (2 * com_norm_std_analytic ** 2)
            - torch.log(com_norms**2 / (math.sqrt(2) * com_norm_std_analytic**3 * scipy.special.gamma(1.5)))
        )

    def sample(self, num_samples: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate num_samples proposals in batches.

        If use_com_adjustment is True, applies a CoM energy adjustment to log_q
        using std = 1/sqrt(num_atoms), as per Prop. 1 of https://arxiv.org/pdf/2502.18462.
        """
        all_samples, all_log_q = [], []
        remaining = num_samples
        while remaining > 0:
            n = min(self.sample_batch_size, remaining)
            s, lq = self.sample_fn(n, **kwargs)
            all_samples.append(s)
            all_log_q.append(lq)
            remaining -= n
        samples = torch.cat(all_samples, dim=0)
        log_q = torch.cat(all_log_q, dim=0)

        if self.use_com_adjustment:
            log_q = log_q + self.com_energy_adjustment(samples)

        return samples, log_q

    @torch.no_grad()
    def energy(self, x: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """Compute energy in batches."""
        bs = batch_size or self.energy_batch_size
        out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        for i in range(0, x.shape[0], bs):
            x_batch = x[i : i + bs]
            out_batch = self.energy_fn(x_batch)
            if self.use_com_adjustment:
                out_batch = out_batch + self.com_energy_adjustment(x_batch)
            out[i : i + bs] = out_batch
        return out

    def energy_and_grad(
        self, x: torch.Tensor, batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and gradient in batches."""
        bs = batch_size or self.grad_batch_size
        e_out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        g_out = torch.empty_like(x)
        for i in range(0, x.shape[0], bs):
            x_batch = x[i : i + bs].detach().requires_grad_(True)
            with torch.enable_grad():
                e_batch = self.energy_fn(x_batch)
                if self.use_com_adjustment:
                    e_batch = e_batch + self.com_energy_adjustment(x_batch)
                g_batch = torch.autograd.grad(e_batch.sum(), x_batch)[0]
            e_out[i : i + bs] = e_batch.detach()
            g_out[i : i + bs] = g_batch.detach()
        return e_out, g_out


@dataclass
class SystemCond:
    permutations: Optional[dict] = None
    encodings: Optional[dict] = None


@dataclass
class EvalContext:
    true_data: "SamplesData"
    target_energy: TargetEnergy
    normalization_std: torch.Tensor
    system_cond: Optional[SystemCond]
    tica_model: Optional[object] = None
    topology: Optional[object] = None


@dataclass
class SamplesData:
    samples: torch.Tensor
    energy: torch.Tensor
    logw: torch.Tensor = None

    def __post_init__(self):
        assert len(self.samples) == len(self.energy)
        if self.logw is not None:
            assert len(self.samples) == len(self.logw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return SamplesData(
            self.samples[index],
            self.energy[index],
            self.logw[index] if self.logw is not None else None,
        )
