"""
Tests for the training pipelines.
NOTE: A very basic test that only checks a single iteration and non-NaN loss.
"""

import os
from math import isnan
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.utils import compose_config, get_config_stem

EXPERIMENT_NAMES = [
    f"training/{cfg_path}"
    for cfg_path in [
        "single_system/ecnf++_Ace-A-Nme.yaml",  # single system ecnf
        "single_system/tarflow_Ace-A-Nme.yaml",  # single system nf
        "transferable/ecnf++_up_to_4aa.yaml",  # transferable ecnf
        "transferable/prose_up_to_8aa.yaml",  # transferable nf
    ]
]


@pytest.fixture(params=EXPERIMENT_NAMES, ids=lambda p: get_config_stem(p), scope="function")
def cfg_test_train_mwe(request: pytest.FixtureRequest, trainer_name_param: str, tmp_path: Path) -> DictConfig:
    """
    Hydra-composed config for the training experiments.

    Args:
        request: pytest request object to get the experiment override parameter.
        trainer_name_param: trainer name parameter supplied by parametrization fixtures.
        tmp_path: pytest-provided temporary directory path.

    Returns:
        DictConfig: Composed and patched Hydra config for the test.
    """
    # Important: clear Hydra before initializing
    GlobalHydra.instance().clear()

    experiment_name = request.param

    cfg = compose_config(
        config_name="train", overrides=[f"experiment={experiment_name}", f"trainer={trainer_name_param}"]
    )

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = os.getcwd()
        cfg.trainer.num_sanity_val_steps = 0  # disable val sanity checks
        cfg.test = False  # disable test stage during training tests
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.data.num_workers = 0  # avoid multiprocessing issues in tests
        cfg.data.batch_size = 32
        cfg.tags = ["pytest", f"test_train_mwe_{trainer_name_param}"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM issues
@pytest.mark.pipeline
def test_train_mwe(cfg_test_train_mwe: DictConfig) -> None:
    """
    Runs train() for every experiment config discovered via the fixture.

    Asserts:
    - 'train/loss' is present in returned metrics and is not NaN.
    """
    metrics, _ = train(cfg_test_train_mwe)

    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"
