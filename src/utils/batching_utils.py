import torch
from typing import Optional, Callable

from src.utils.dataclasses import ProposalModel

def tree_map(fn, tree):
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    elif isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, (tuple, list)):
        return type(tree)(tree_map(fn, v) for v in tree)
    else:
        return tree  # unchanged non-tensors

def tree_cat(tree_list):
    """Concatenate a list of trees along batch dimension.

    Each element must share the same nested structure.
    """

    assert len(tree_list) > 0, "tree_cat() called with an empty list"

    elem0 = tree_list[0]

    # Case 1: Tensor leaves
    if isinstance(elem0, torch.Tensor):
        # dtype consistency
        for t in tree_list:
            assert isinstance(t, torch.Tensor), "Mismatch: some leaves are tensors, others are not"
            assert t.shape[1:] == elem0.shape[1:], (
                f"Tensor shapes differ after batch dim: {t.shape} vs {elem0.shape}"
            )
            assert t.dtype == elem0.dtype, "Tensor dtype mismatch"
            assert t.device == elem0.device, "Tensor device mismatch"

        return torch.cat(tree_list, dim=0)

    # Case 2: dict
    elif isinstance(elem0, dict):
        keys = set(elem0.keys())
        for t in tree_list:
            assert isinstance(t, dict), "Mismatch: some leaves are dicts, others are not"
            assert set(t.keys()) == keys, f"Dict keys differ: {set(t.keys())} vs {keys}"

        return {k: tree_cat([t[k] for t in tree_list]) for k in keys}

    # Case 3: tuple or list
    elif isinstance(elem0, (list, tuple)):
        length = len(elem0)
        for t in tree_list:
            assert isinstance(t, type(elem0)), "List/tuple type mismatch"
            assert len(t) == length, f"Length mismatch: {len(t)} vs {length}"

        return type(elem0)(
            tree_cat(items) for items in zip(*tree_list)
        )

    # Case 4: default — must be identical
    else:
        for t in tree_list:
            assert t == elem0, (
                f"Non-tensor leaves must be identical, got {t} vs {elem0}"
            )
        return elem0

def repeat_for_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)


def batchify_system_cond(system_cond, batch_size: int, device=None):
    result = tree_map(
        lambda t: repeat_for_batch(t, batch_size) if isinstance(t, torch.Tensor) else t,
        system_cond
    )
    if device is not None:
        result = tree_map(
            lambda t: t.to(device) if isinstance(t, torch.Tensor) else t,
            result
        )
    return result


def flatten_ddp_gather(tensor: torch.Tensor) -> torch.Tensor:
    # Lightning all_gather returns shape: (world_size, batch, ...)
    # We flatten world_size * batch into a single batch dimension
    return tensor.reshape(-1, *tensor.shape[2:])


def gather_tree(tree, gather_fn):
    if gather_fn is None:
        return tree
    return tree_map(lambda t: flatten_ddp_gather(gather_fn(t)), tree)


def batchify(
    fn,
    batch_size: int,
    *,
    gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    """
    Returns a wrapper around fn that splits input `x` into batches,
    applies fn to each batch, optionally gathers across DDP workers,
    and concatenates the results.

    Supports nested structures of tensors (dict/tuple/list).
    """

    def apply_batchify(x, *args, **kwargs):
        outputs = []
        B = len(x)

        for start in range(0, B, batch_size):
            end = start + batch_size
            x_chunk = x[start:end]

            out = fn(x_chunk, *args, **kwargs)

            # optionally gather across DDP workers
            if gather is not None:
                # final shape: (world_size * batch, ...)
                out = tree_map(gather, out)
                out = tree_map(flatten_ddp_gather, out)

            outputs.append(out)

        # combine chunk results
        return tree_cat(outputs)

    return apply_batchify


def generate_in_batches(proposal_model: ProposalModel, num_total: int, batch_size: int, system_cond: Optional[dict[str, torch.Tensor]] = None) -> torch.Tensor:
    batches = []
    for _ in range(num_total // batch_size):
        batches.append(proposal_model.sample(batch_size, system_cond))
    if num_total % batch_size != 0:
        batches.append(proposal_model.sample(num_total % batch_size, system_cond))
    return tree_cat(batches)