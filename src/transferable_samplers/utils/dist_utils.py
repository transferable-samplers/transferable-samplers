from __future__ import annotations

from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def is_distributed() -> bool:
    """Check if torch distributed is initialized and has more than 1 rank."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_world_size() -> int:
    """Return the distributed world size, or 1 if not distributed."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Return the current process rank, or 0 if not distributed."""
    return dist.get_rank() if dist.is_initialized() else 0


def drift_rng_state(stride: int = 10_000) -> None:
    """Desync the per-rank RNG by consuming `rank * stride` random draws.

    Lightning's ``seed_everything`` sets the same seed on every rank, which
    causes rank-local sampling ops (e.g. ``source_energy.sample`` drawing
    from a normalising-flow base distribution) to produce identical outputs
    across ranks. Calling this once after DDP is initialised consumes a
    rank-dependent number of random draws so subsequent draws decorrelate.
    No-op on rank 0 / non-distributed.

    Drifts CPU and (if available) the current CUDA device's RNG. PyTorch's
    CPU and CUDA RNGs are independent generators, and code paths that look
    GPU-only often sample on CPU then move to GPU (e.g.
    ``torch.distributions.Normal`` with scalar params), so we have to
    advance both to be safe.
    """
    rank = get_rank()
    if rank == 0:
        return
    n = rank * stride
    torch.randn(n)
    if torch.cuda.is_available():
        torch.randn(n, device=torch.cuda.current_device())


def all_gather_cat(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor across ranks and concatenate along dim 0.

    Safe for non-distributed settings — returns the tensor unchanged.
    """
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from src rank to all ranks.

    Safe for non-distributed settings — returns the tensor unchanged.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def shard_tensor(x: T) -> T:
    """Shard x across DDP ranks, asserts number of particles exactly divisible by world size.

    Works for any object supporting len() and slicing (tensors, SMCParticles, etc.).
    """
    world_size = get_world_size()
    rank = get_rank()
    assert len(x) % world_size == 0, (
        f"Length of x ({len(x)}) must be divisible by world size ({world_size}) for sharding."
    )
    n = (len(x) // world_size) * world_size
    chunk_size = n // world_size
    return x[rank * chunk_size : (rank + 1) * chunk_size]
