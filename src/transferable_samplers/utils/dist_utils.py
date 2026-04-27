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


def drift_rng_state() -> None:
    """Desync the per-rank RNG by seeking each rank's generators to a unique offset.

    Lightning's ``seed_everything`` sets the same seed on every rank, which
    causes rank-local sampling ops (e.g. ``source_energy.sample`` drawing
    from a normalising-flow base distribution) to produce identical outputs
    across ranks. Calling this once after DDP is initialised advances each
    rank's RNG by ``rank * STRIDE``, so subsequent draws decorrelate.
    No-op on rank 0 / non-distributed.

    Drifts CPU and (if available) the current CUDA device's RNG. PyTorch's
    CPU and CUDA RNGs are independent generators, and code paths that look
    GPU-only often sample on CPU then move to GPU (e.g.
    ``torch.distributions.Normal`` with scalar params), so we have to
    advance both.

    CUDA uses a counter-based Philox generator, so we seek directly via
    ``Generator.set_offset`` rather than burning kernel launches — note that
    ``torch.randn(N, device=cuda)`` advances the offset by a fixed amount
    *per launch*, not per element, so a single big call would not decorrelate
    ranks. CPU's Mersenne Twister has no cheap seek, but it advances per
    element, so a single ``torch.randn(rank * STRIDE)`` is fine.

    STRIDE is chosen so each rank gets a sub-stream large enough to never
    collide in practice: ~10^6 launches/elements per rank gap, well above
    any realistic per-rank RNG consumption in a single run, and a tiny
    fraction of Philox's 2^64 offset space.
    """
    STRIDE = 1 << 20
    rank = get_rank()
    if rank == 0:
        return
    torch.randn(rank * STRIDE)
    if torch.cuda.is_available():
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        gen.set_offset(gen.get_offset() + rank * STRIDE)


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
