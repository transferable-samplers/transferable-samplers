import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if torch distributed is initialized and has more than 1 rank."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


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


def shard_tensor(x):
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
