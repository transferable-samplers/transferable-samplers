"""
If this test collection passes, we know that training works for both NF and ECNF models on
both single-system and transferable settings.
"""

import os
from math import isnan
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.train import train

# Locate relevant experiment config files
CONFIG_BASE = Path(__file__).resolve().parent.parent / "configs"
TRAIN_DIR = CONFIG_BASE / "experiment" / "training"
EXPERIMENT_CONFIGS = [
    TRAIN_DIR / Path(p)
    for p in [
        "single_system/ecnf++_Ace-A-Nme.yaml",  # single system ecnf
        "single_system/tarflow_Ace-A-Nme.yaml",  # single system nf
        "transferable/ecnf++_up_to_4aa.yaml",  # transferable ecnf
        "transferable/prose_up_to_8aa.yaml",  # transferable nf
    ]
]


@pytest.fixture(params=EXPERIMENT_CONFIGS, ids=lambda p: p.stem, scope="function")
def cfg_test_train(request: pytest.FixtureRequest, tmp_path: Path) -> DictConfig:
    """
    Parameterized Hydra-composed config for each experiment file under configs/experiment/evaluation.
    Each test that takes `cfg_eval_param` will run once per config file.

    This fixture safely resets Hydra between runs and patches all paths.
    """
    cfg_path: Path = request.param
    rel_path = cfg_path.relative_to(CONFIG_BASE).with_suffix("")
    override = rel_path.as_posix().removeprefix("experiment/")

    # Important: clear Hydra before initializing
    GlobalHydra.instance().clear()

    # Compose full Hydra config
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=[f"experiment={override}"])

    # Patch common paths to avoid writing to the project tree
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = os.getcwd()
        cfg.trainer.num_sanity_val_steps = 0  # disable val sanity checks
        cfg.test = False  # disable test stage during training tests
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.data.batch_size = 32
        cfg.tags = ["pytest", "test_train"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.slow
def test_train(cfg_test_train: DictConfig):
    """
    Runs eval() for every experiment config discovered via the fixture.
    :param cfg_eval_param: The configuration for the evaluation.
    """

    metrics, _ = train(cfg_test_train)

    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"
