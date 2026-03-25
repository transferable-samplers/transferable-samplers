from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import scipy.special
import torch
import torch.utils._pytree as pytree

from transferable_samplers.utils.pylogger import RankedLogger
from transferable_samplers.utils.standardization import destandardize_coords

logger = RankedLogger(__name__, rank_zero_only=False)


@dataclass
class TargetEnergy:
    """Target energy function, provided by the datamodule.

    Handles unnormalization internally: callers pass normalized samples,
    and the energy function receives unnormalized positions.
    """

    energy_fn: Callable  # (x_unnormalized) -> energy (batch,)
    normalization_std: torch.Tensor

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute target energy on normalized samples.

        Destandardizes coordinates internally before evaluating.
        """
        return self.energy_fn(destandardize_coords(x, self.normalization_std))

    def energy_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute target energy and its gradient on normalized samples.

        Must be called outside ``torch.enable_grad()`` context. Internally
        enables gradients to compute the gradient via autograd, then detaches.

        Raises:
            RuntimeError: If called while gradients are globally enabled.
        """
        if torch.is_grad_enabled():
            raise RuntimeError(
                "TargetEnergy.energy_and_grad() is non-differentiable by design."
                "It is not implemented to get higher-order terms from OpenMM."
                "Use TargetEnergy.energy() for differentiable energy evaluations."
            )
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            e = self.energy_fn(destandardize_coords(x, self.normalization_std))
            g = torch.autograd.grad(e.sum(), x)[0]
        return e.detach(), g.detach()


@dataclass(frozen=True)
class SourceEnergyConfig:
    """Configuration for SourceEnergy batch sizes. Passed via Hydra configs.

    All batch sizes are global (across all devices), following Lightning convention.
    SourceEnergy.__post_init__ divides by world_size to get per-device batch sizes.
    """

    sample_batch_size: int
    energy_batch_size: int
    grad_batch_size: int
    use_com_adjustment: bool = False


@dataclass
class SourceEnergy:
    """Source (proposal) energy and sampling, provided by the lightning module.

    Created via BaseLightningModule.build_source_energy(). Handles batching
    internally for sample, energy, and energy_and_grad. Batch sizes are global
    (following Lightning convention) and divided by world_size in __post_init__.
    """

    sample_fn: Callable  # (num_samples) -> (samples, E_source)
    energy_fn: Callable  # (x) -> energy (batch,)
    sample_batch_size: int
    energy_batch_size: int
    grad_batch_size: int
    use_com_adjustment: bool = False
    world_size: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        from transferable_samplers.utils.dist_utils import get_world_size

        self.world_size = get_world_size()
        assert self.sample_batch_size % self.world_size == 0, (
            f"sample_batch_size ({self.sample_batch_size}) must be divisible by world_size ({self.world_size})"
        )
        assert self.energy_batch_size % self.world_size == 0, (
            f"energy_batch_size ({self.energy_batch_size}) must be divisible by world_size ({self.world_size})"
        )
        assert self.grad_batch_size % self.world_size == 0, (
            f"grad_batch_size ({self.grad_batch_size}) must be divisible by world_size ({self.world_size})"
        )
        self.sample_batch_size = self.sample_batch_size // self.world_size
        self.energy_batch_size = self.energy_batch_size // self.world_size
        self.grad_batch_size = self.grad_batch_size // self.world_size

    @staticmethod
    def com_energy_adjustment(x: torch.Tensor) -> torch.Tensor:
        """CoM energy adjustment to add to from E_source (= subtract from log q).

        Introduced in Prop. 1 of https://arxiv.org/pdf/2502.18462.
        x: (batch, num_atoms, 3) -> energy adjustment (batch,) to add to E_source.

        NOTE: need to benchmark / implement the fixed version from https://arxiv.org/pdf/2602.03729
        """
        # The mean of N i.i.d. standard Gaussian samples has std = 1/sqrt(N)
        # This is the same as used in the data augmentation.
        com_std_analytic = 1.0 / math.sqrt(x.shape[1])

        # Compute the Norms of the CoM for each sample in the batch.
        com_norms = x.mean(dim=1).norm(dim=-1)

        return -(com_norms**2) / (2 * com_std_analytic**2) + torch.log(
            com_norms**2 / (math.sqrt(2) * com_std_analytic**3 * scipy.special.gamma(1.5))
        )

    def sample(self, num_samples: int, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate num_samples proposals in batches on this device.

        num_samples is per-device: each device independently generates num_samples,
        and the caller is responsible for all_gather if a global view is needed.

        If use_com_adjustment is True, applies a CoM energy adjustment
        using std = 1/sqrt(num_atoms), as per Prop. 1 of https://arxiv.org/pdf/2502.18462.
        """
        num_batches = math.ceil(num_samples / self.sample_batch_size)
        logger.info(
            f"Sampling {num_batches} batches of {self.sample_batch_size} samples per device, "
            f"total {num_samples} samples per device {num_samples * self.world_size} samples."
        )

        all_samples, all_E = [], []
        remaining = num_samples
        batch_idx = 0
        while remaining > 0:
            n = min(self.sample_batch_size, remaining)
            t0 = time.perf_counter()
            s, e = self.sample_fn(n, **kwargs)
            elapsed = time.perf_counter() - t0
            throughput = n / elapsed if elapsed > 0 else float("inf")
            batch_idx += 1
            logger.info(f"Proposal sampling batch {batch_idx}/{num_batches} @ {throughput:.1f} samples/s")
            all_samples.append(s)
            all_E.append(e)
            remaining -= n

        samples = torch.cat(all_samples, dim=0)
        E_source = torch.cat(all_E, dim=0)

        if self.use_com_adjustment:
            E_source = E_source + self.com_energy_adjustment(samples)

        return samples, E_source

    def energy(self, x: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """Compute energy in batches on this device.

        x is per-device: each device independently evaluates its local samples.
        The caller is responsible for all_gather if a global view is needed.

        If use_com_adjustment is True, applies a CoM energy adjustment
        using std = 1/sqrt(num_atoms), as per Prop. 1 of https://arxiv.org/pdf/2502.18462.
        """
        bs = batch_size or self.energy_batch_size
        out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        for i in range(0, x.shape[0], bs):
            x_batch = x[i : i + bs]
            out_batch = self.energy_fn(x_batch)
            if self.use_com_adjustment:
                out_batch = out_batch + self.com_energy_adjustment(x_batch)
            out[i : i + bs] = out_batch
        return out

    def energy_and_grad(self, x: torch.Tensor, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and gradient in batches on this device.

        x is per-device: each device independently evaluates its local samples.
        The caller is responsible for all_gather if a global view is needed.

        If use_com_adjustment is True, applies a CoM energy adjustment
        using std = 1/sqrt(num_atoms), as per Prop. 1 of https://arxiv.org/pdf/2502.18462.
        """
        if torch.is_grad_enabled():
            raise RuntimeError(
                "SourceEnergy.energy_and_grad() is non-differentiable by convention "
                "with TargetEnergy.energy_and_grad(). "
                "Use SourceEnergy.energy() for differentiable energy evaluations. "
                "Take care to test well if removing this!"
            )
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
    """System conditioning (permutations, encodings) for transferable models.

    Stores unbatched conditioning tensors. Use for_batch() to expand to batch size.
    """

    permutations: dict[str, torch.Tensor] | None = None
    encodings: dict[str, torch.Tensor] | None = None

    def for_batch(self, batch_size: int, device: torch.device | None = None) -> SystemCond:
        """Expand unbatched conditioning tensors to batch_size and move to device."""

        def expand(v: torch.Tensor) -> torch.Tensor:
            v = v.unsqueeze(0).expand(batch_size, *v.shape)
            return v.to(device) if device is not None else v

        return SystemCond(
            encodings=pytree.tree_map(expand, self.encodings) if self.encodings else None,
            permutations=pytree.tree_map(expand, self.permutations) if self.permutations else None,
        )


@dataclass
class SamplesData:
    """Container for sampled conformations to be passed to the evaluator."""

    samples: torch.Tensor
    E_target: torch.Tensor
    logw: torch.Tensor | None = None

    def __post_init__(self) -> None:
        assert len(self.samples) == len(self.E_target)
        if self.logw is not None:
            assert len(self.samples) == len(self.logw)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | slice | torch.Tensor) -> SamplesData:
        return SamplesData(
            self.samples[index],
            self.E_target[index],
            self.logw[index] if self.logw is not None else None,
        )


@dataclass
class EvalContext:
    """Evaluation context bundling reference data, energy functions, and metadata."""

    true_data: SamplesData
    target_energy: TargetEnergy
    normalization_std: torch.Tensor
    system_cond: SystemCond | None
    tica_model: object | None = None
    topology: object | None = None
