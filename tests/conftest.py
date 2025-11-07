"""This file prepares config fixtures for other tests."""

import os
from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg

@pytest.fixture(scope="function")
def cfg_eval(tmp_path: Path) -> DictConfig:
    """Hydra-composed eval config for a specific experiment."""

    experiment_rel = "evaluation/transferable/prose_up_to_8aa_is"

    # Important: clear any existing Hydra instance (especially across tests)
    GlobalHydra.instance().clear()

    # Compose full config
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", overrides=[f"experiment={experiment_rel}"])

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = os.getcwd()
    
    yield cfg

    GlobalHydra.instance().clear()
