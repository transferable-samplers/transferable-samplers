from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import filelock
import torch

from transferable_samplers.utils.huggingface import download_weights
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


def load_state_dict_from_ckpt(path: str) -> dict[str, torch.Tensor]:
    """Extract a state dict from a Lightning checkpoint file (``state_dict`` key).

    Args:
        path: Path to the Lightning checkpoint file.

    Returns:
        The state dict extracted from the checkpoint.

    Raises:
        KeyError: If the checkpoint does not contain a ``state_dict`` key.

    NOTE: If the model was trained using EMAWeightAveraging, the EMA weights
    will be in the checkpoint's ``state_dict`` field.
    """
    assert path.endswith(".ckpt"), f"Expected a .ckpt file, got: {path}"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint at {path} missing 'state_dict' key.")
    return ckpt["state_dict"]


def load_state_dict_from_file(path: str) -> dict[str, torch.Tensor]:
    """Load a raw state dict from a ``.pth`` file.

    Args:
        path: Path to the state-dict file.

    Returns:
        The loaded state dict.
    """
    assert path.endswith(".pth"), f"Expected a .pth file, got: {path}"
    return torch.load(path, map_location="cpu", weights_only=False)


def load_state_dict_from_hf(hf_filepath: str, scratch_dir: str) -> dict[str, torch.Tensor]:
    """Download a state dict from HuggingFace Hub and return it.

    Uses a file lock so concurrent DDP processes don't race on the download.
    ``dist`` is not yet initialized when this is called (it's called before
    ``trainer.test/fit``), so we use a filesystem lock instead of a barrier.

    Args:
        hf_filepath: The filepath within the HuggingFace repo.
        scratch_dir: Local scratch directory used to cache downloaded weights.

    Returns:
        The loaded state dict.
    """
    dst_dir = str(Path(scratch_dir) / "model-weights")
    lock_path = str(Path(scratch_dir) / ".hf_download.lock")
    with filelock.FileLock(lock_path):
        local_path = download_weights(hf_filepath=hf_filepath, destination_dir=dst_dir)
    return load_state_dict_from_file(local_path)


def augment_state_dict_for_teacher(
    sd: dict[str, torch.Tensor],
    student_prefix: str = "net.",
    teacher_prefix: str = "teacher.",
) -> dict[str, torch.Tensor]:
    """Synthesize teacher keys by copying from student keys if missing.

    Returns a NEW dict.
    """
    has_student = any(k.startswith(student_prefix) for k in sd.keys())
    assert has_student, "Expected student keys in state dict when calling augment_state_dict_for_teacher."

    has_teacher = any(k.startswith(teacher_prefix) for k in sd.keys())
    assert not has_teacher, "Expected no teacher keys in state dict when calling augment_state_dict_for_teacher."

    logger.info("Augmenting state dict for teacher by copying student parameters.")

    new_sd = deepcopy(sd)
    for k, v in sd.items():
        if k.startswith(student_prefix):
            teacher_k = k.replace(student_prefix, teacher_prefix, 1)
            new_sd[teacher_k] = v
    return new_sd


def resolve_init(
    init_ckpt_path: str | None,
    init_hf_state_dict_path: str | None,
    scratch_dir: str,
) -> dict[str, torch.Tensor] | None:
    """Resolve init weights from a checkpoint or HuggingFace state dict.

    Args:
        init_ckpt_path: Path to a Lightning checkpoint for weights-only init.
        init_hf_state_dict_path: HuggingFace filepath for weights-only init.
        scratch_dir: Local scratch directory for caching HF downloads.

    Returns:
        The init state dict, or ``None`` if no init source was provided.

    Raises:
        AssertionError: If both ``init_ckpt_path`` and ``init_hf_state_dict_path``
            are set (they are mutually exclusive).
        FileNotFoundError: If ``init_ckpt_path`` is set but the file does not exist.
    """
    assert not (init_ckpt_path and init_hf_state_dict_path), (
        "At most one of init_ckpt_path / init_hf_state_dict_path may be set."
    )

    if init_hf_state_dict_path is not None:
        logger.info("Downloading init weights from HuggingFace...")
        state_dict = load_state_dict_from_hf(init_hf_state_dict_path, scratch_dir)
        logger.info("Loaded init weights from HuggingFace state_dict.")
        return state_dict

    if init_ckpt_path is not None:
        if not Path(init_ckpt_path).exists():
            raise FileNotFoundError(f"init_ckpt_path not found: {init_ckpt_path}")
        logger.info(f"Loading init weights from checkpoint: {init_ckpt_path}")
        state_dict = load_state_dict_from_ckpt(init_ckpt_path)
        logger.info("Loaded init weights from init checkpoint.")
        return state_dict

    return None


def resolve_init_or_resume(
    resume_ckpt_path: str | None,
    init_ckpt_path: str | None,
    init_hf_state_dict_path: str | None,
    scratch_dir: str,
) -> tuple[str | None, dict[str, torch.Tensor] | None]:
    """Determine whether to resume from a checkpoint or apply init weights.

    Implements preemptible semantics:
    - If ``resume_ckpt_path`` exists on disk and is loadable, resume from it
      (ignoring any init weights).
    - Otherwise, resolve init weights if provided, or start from scratch.

    Args:
        resume_ckpt_path: Path for full Lightning resume. May not exist yet.
        init_ckpt_path: Path to a Lightning checkpoint for weights-only init.
        init_hf_state_dict_path: HuggingFace filepath for weights-only init.
        scratch_dir: Local scratch directory for caching HF downloads.

    Returns:
        A tuple of ``(fit_ckpt_path, init_state_dict)`` where exactly one (or
        neither) is non-None:
        - ``fit_ckpt_path``: checkpoint path to pass to ``trainer.fit(ckpt_path=...)``.
        - ``init_state_dict``: state dict to load into the model before training.

    Raises:
        AssertionError: If both ``init_ckpt_path`` and ``init_hf_state_dict_path``
            are set (they are mutually exclusive).
        FileNotFoundError: If ``init_ckpt_path`` is set but the file does not exist.
    """
    # 1. Try to resume from an existing checkpoint
    if resume_ckpt_path and Path(resume_ckpt_path).exists():
        try:
            _ = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
            logger.info(f"Found resume checkpoint at {resume_ckpt_path}. Resuming training; ignoring init_*.")
            return resume_ckpt_path, None
        except Exception:
            logger.exception(
                f"Resume checkpoint exists but could not be loaded: {resume_ckpt_path}. Falling back to init/scratch."
            )

    # 2. No valid resume checkpoint — resolve init weights or start from scratch
    if resume_ckpt_path and not Path(resume_ckpt_path).exists():
        logger.warning(f"resume_ckpt_path set but not found: {resume_ckpt_path}. ")

    init_state_dict = resolve_init(init_ckpt_path, init_hf_state_dict_path, scratch_dir)

    if init_state_dict is None:
        logger.info("No resume checkpoint found and no init weights provided; training from random initialization.")

    return None, init_state_dict
