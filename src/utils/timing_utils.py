import time
import torch
import contextlib

# -------------------------------------------------------
# Safe helpers for distributed sync
# -------------------------------------------------------

def cuda_sync():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def distributed_sync():
    """
    Synchronize all ranks only if torch.distributed is initialized.
    Safe to call in CPU/single-GPU/notebook mode.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
        except RuntimeError:
            # Happens if backend not properly set up; fail silent
            pass


# -------------------------------------------------------
# Timing context manager
# -------------------------------------------------------

@contextlib.contextmanager
def timed_block(sync_cuda: bool = True, sync_dist: bool = False):
    """
    Context manager for accurately measuring wall-clock time of GPU operations
    or distributed-synchronized operations.

    Args:
        sync_cuda (bool):
            If True, call cuda.synchronize() before and after timing block.
        sync_dist (bool):
            If True, call distributed barrier before and after timing block.

    Usage:
        with timed_block(sync_cuda=True, sync_dist=True) as t:
            something()
        print(t.elapsed)
    """
    if sync_cuda:
        cuda_sync()
    if sync_dist:
        distributed_sync()

    start = time.time()

    yield type("TimingResult", (), {"start": start})

    if sync_cuda:
        cuda_sync()
    if sync_dist:
        distributed_sync()

    end = time.time()
    setattr(_, "elapsed", end - start)  # attach elapsed time dynamically


def timing_metrics(elapsed: float, n: int, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}/walltime": elapsed,
        f"{prefix}/num_per_second": n / elapsed,
        f"{prefix}/seconds_per_item": elapsed / n,
    }