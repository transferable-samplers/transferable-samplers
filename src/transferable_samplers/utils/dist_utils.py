from __future__ import annotations

import os
import random
from typing import TypeVar

import numpy as np
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


def seed_rank_local() -> None:
    """Re-seed this rank's RNG with a rank-unique seed derived from the base seed.

    Lightning's ``seed_everything`` sets the same seed on every rank, which
    causes rank-local sampling ops (e.g. ``source_energy.sample`` drawing
    from a normalising-flow base distribution) to produce identical outputs
    across ranks. This derives a unique seed per rank by offsetting the base
    seed (read from ``PL_GLOBAL_SEED``, set by ``seed_everything``) with the
    rank, then re-seeds torch, numpy, and python.random.

    Unlike offset-based RNG drift, per-rank seeds produce fully independent
    streams (no collision risk), cover all three RNG sources (torch, numpy,
    python.random), and are idempotent — calling this multiple times from
    repeated ``setup()`` calls produces the same state (no compounding).

    No-op on rank 0 (already correctly seeded by ``seed_everything``) or
    when distributed is not initialised.

    Note: ``PL_GLOBAL_SEED`` is intentionally not overwritten so that
    subsequent calls remain idempotent and DataLoader worker seeding
    (``pl_worker_init_function``, which derives from ``torch.initial_seed()``
    + rank + worker_id) automatically picks up the per-rank seed.
    """
    rank = get_rank()
    if rank == 0:
        return
    base_seed = int(os.environ.get("PL_GLOBAL_SEED", "0"))
    per_rank_seed = base_seed + rank
    torch.manual_seed(per_rank_seed)
    np.random.seed(per_rank_seed)
    random.seed(per_rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(per_rank_seed)


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
