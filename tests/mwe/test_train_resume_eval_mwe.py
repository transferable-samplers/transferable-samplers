"""
Integration tests for train+resume, train-from-init, and train+eval workflows.

These tests exercise the checkpoint / init-weight machinery end-to-end:
  1. train + resume   — train 1 epoch, resume from last.ckpt for 1 more epoch
  2. train from init  — train to produce a checkpoint, then re-init from:
       a. ckpt_path  (Lightning .ckpt → state_dict)
       b. hf_state_dict (raw .pth file, simulating HF download)
     Both variants also test resume on top of init.
  3. train + eval     — train 1 epoch, then eval (test stage) from last.ckpt
"""

from math import isnan
from pathlib import Path

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config
from transferable_samplers.train import train

# PyTorch 2.6+ defaults to weights_only=True. Lightning checkpoints contain
# OmegaConf objects so we need weights_only=False for resume to work.
# Patch torch.load so None→False (pre-2.6 behaviour).
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load  # type: ignore[assignment]

# Use a single lightweight experiment config for all tests.
EXPERIMENT = "single_system/train/tarflow_AAA.yaml"


def _make_train_cfg(trainer_name: str, tmp_path: Path, **extra) -> DictConfig:
    """Compose a minimal training config for testing."""
    GlobalHydra.instance().clear()
    cfg = compose_config(
        config_name="train",
        overrides=[f"experiment={EXPERIMENT}", f"trainer={trainer_name}"],
    )
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.data.num_workers = 0
        cfg.data.batch_size = 2
        cfg.seed = 42
        # Shrink network for faster CI runs
        cfg.model.net.channels = 64
        cfg.model.net.num_blocks = 2
        cfg.model.net.layers_per_block = 2
        cfg.tags = ["pytest"]
        # Disable sampling evaluation callback (not needed for training-only tests).
        cfg.callbacks.sampling_evaluation = None
        # Disable LR scheduler to avoid OneCycleLR total_steps issues on resume.
        cfg.model.scheduler = None
        for k, v in extra.items():
            OmegaConf.update(cfg, k, v)
    return cfg


def _assert_train_loss(metrics: dict) -> None:
    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"


def _train_and_get_ckpt(trainer_name: str, out_dir: Path, **extra) -> Path:
    """Train 1 epoch and return the path to last.ckpt."""
    cfg = _make_train_cfg(trainer_name, out_dir, **extra)
    metrics, _ = train(cfg)
    _assert_train_loss(metrics)
    last_ckpt = out_dir / "checkpoints" / "last.ckpt"
    assert last_ckpt.exists(), f"Expected last.ckpt at {last_ckpt}"
    return last_ckpt


# ---------------------------------------------------------------------------
# 1. Train + Resume
# ---------------------------------------------------------------------------
@pytest.mark.essential
def test_train_and_resume(trainer_name_param: str, tmp_path: Path) -> None:
    """Train 1 epoch, then resume from last.ckpt for 1 more epoch."""
    # --- initial training run ---
    last_ckpt = _train_and_get_ckpt(trainer_name_param, tmp_path)

    # --- resumed training run (continue to epoch 2) ---
    resume_dir = tmp_path / "resume_run"
    resume_dir.mkdir()
    cfg2 = _make_train_cfg(
        trainer_name_param,
        resume_dir,
        resume_ckpt_path=str(last_ckpt),
        **{"trainer.max_epochs": 2},
    )
    metrics2, _ = train(cfg2)
    _assert_train_loss(metrics2)

    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# 2a. Train from init: ckpt_path
# ---------------------------------------------------------------------------
@pytest.mark.essential
def test_train_from_init_ckpt(trainer_name_param: str, tmp_path: Path) -> None:
    """Train to create a checkpoint, then init a fresh run from that ckpt."""
    seed_dir = tmp_path / "seed_run"
    seed_dir.mkdir()
    last_ckpt = _train_and_get_ckpt(trainer_name_param, seed_dir)

    # --- train from init ckpt_path ---
    init_dir = tmp_path / "init_ckpt_run"
    init_dir.mkdir()
    cfg_init = _make_train_cfg(trainer_name_param, init_dir, init_ckpt_path=str(last_ckpt))
    metrics, _ = train(cfg_init)
    _assert_train_loss(metrics)

    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# 2b. Train from init: ckpt_path + resume
# ---------------------------------------------------------------------------
@pytest.mark.essential
def test_train_from_init_ckpt_then_resume(trainer_name_param: str, tmp_path: Path) -> None:
    """Init from ckpt, train 1 epoch, then resume for 1 more epoch."""
    seed_dir = tmp_path / "seed_run"
    seed_dir.mkdir()
    seed_ckpt = _train_and_get_ckpt(trainer_name_param, seed_dir)

    # --- train from init ---
    init_dir = tmp_path / "init_run"
    init_dir.mkdir()
    init_ckpt = _train_and_get_ckpt(trainer_name_param, init_dir, init_ckpt_path=str(seed_ckpt))

    # --- resume on top of init ---
    # When resume_ckpt_path exists, init_ckpt_path is ignored (resume takes precedence).
    resume_dir = tmp_path / "resume_run"
    resume_dir.mkdir()
    cfg_resume = _make_train_cfg(
        trainer_name_param,
        resume_dir,
        init_ckpt_path=str(seed_ckpt),
        resume_ckpt_path=str(init_ckpt),
        **{"trainer.max_epochs": 2},
    )
    metrics, _ = train(cfg_resume)
    _assert_train_loss(metrics)

    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# 2c. Train from init: hf_state_dict (finetune from pretrained HF weights)
# ---------------------------------------------------------------------------
@pytest.mark.essential
def test_train_from_init_hf_state_dict(trainer_name_param: str, tmp_path: Path) -> None:
    """Init from pretrained HF weights and finetune for 1 epoch."""
    cfg = _make_train_cfg(
        trainer_name_param,
        tmp_path,
        init_hf_state_dict_path="single_system/tarflow_AAA_0.pth",
    )
    metrics, _ = train(cfg)
    _assert_train_loss(metrics)

    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# 2d. Train from init: hf_state_dict + resume
# ---------------------------------------------------------------------------
@pytest.mark.essential
def test_train_from_init_hf_state_dict_then_resume(trainer_name_param: str, tmp_path: Path) -> None:
    """Init from pretrained HF weights, train 1 epoch, then resume for 1 more."""
    init_dir = tmp_path / "init_run"
    init_dir.mkdir()
    init_ckpt = _train_and_get_ckpt(
        trainer_name_param, init_dir, init_hf_state_dict_path="single_system/tarflow_AAA_0.pth"
    )

    # --- resume on top of init ---
    resume_dir = tmp_path / "resume_run"
    resume_dir.mkdir()
    cfg_resume = _make_train_cfg(
        trainer_name_param,
        resume_dir,
        resume_ckpt_path=str(init_ckpt),
        **{"trainer.max_epochs": 2},
    )
    metrics, _ = train(cfg_resume)
    _assert_train_loss(metrics)

    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# 3. Train + Eval
# ---------------------------------------------------------------------------
@pytest.mark.essential
def test_train_and_eval(trainer_name_param: str, tmp_path: Path) -> None:
    """Train 1 epoch, then run eval (test stage) loading from last.ckpt."""
    from transferable_samplers.eval import eval

    train_dir = tmp_path / "train_run"
    train_dir.mkdir()
    last_ckpt = _train_and_get_ckpt(trainer_name_param, train_dir)

    # --- eval run from the checkpoint ---
    eval_dir = tmp_path / "eval_run"
    eval_dir.mkdir()

    cfg_eval = compose_config(
        config_name="eval",
        overrides=[
            "experiment=single_system/eval/tarflow_AAA_ula.yaml",
            f"trainer={trainer_name_param}",
        ],
    )
    with open_dict(cfg_eval):
        cfg_eval.paths.output_dir = str(eval_dir)
        cfg_eval.paths.log_dir = str(eval_dir)
        cfg_eval.paths.work_dir = str(Path.cwd())
        cfg_eval.ckpt_path = str(last_ckpt)
        cfg_eval.hf_state_dict_path = None
        cfg_eval.data.num_workers = 0
        cfg_eval.data.batch_size = 4
        # Replace SMC sampler with SNIS for speed in tests.
        cfg_eval.callbacks.sampling_evaluation.sampler = OmegaConf.create(
            {
                "_target_": "transferable_samplers.samplers.snis_sampler.SNISSampler",
                "num_samples": 32,
            }
        )
        if trainer_name_param == "cpu":
            cfg_eval.callbacks.sampling_evaluation.run_diagnostics_kwargs = {
                "num_samples_invert": 8,
                "num_samples_dlogp": 2,
            }
        cfg_eval.tags = ["pytest", "test_train_and_eval"]

    # pyrefly: ignore [bad-argument-type]
    metrics_eval, _ = eval(cfg_eval)

    # The eval should produce proposal energy metrics for AAA.
    key = "test/AAA/proposal/median-energy"
    assert key in metrics_eval, f"{key} missing from eval metrics"

    GlobalHydra.instance().clear()
