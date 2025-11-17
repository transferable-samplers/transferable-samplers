"""
If this test collection passes, we know the self-improvement pipeline can be run end-to-end.
NOTE: Currently there is a hard-coded path to a checkpoint in this test, which needs to be
updated to point to a valid checkpoint on your system.
The entire self-improvement pipeline needs to be reworked to run from huggingface weights,
but this is outside the scope of establishing the test suite.
"""

import os
from math import isnan
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.self_improve import self_improve

# Locate relevant experiment config files
CONFIG_BASE = Path(__file__).resolve().parent.parent / "configs"
EVAL_DIR = CONFIG_BASE / "experiment" / "evaluation"
EXPERIMENT_CONFIG = EVAL_DIR / Path("transferable/prose_up_to_8aa_self_improvement.yaml")
# TODO we need to setup self_improve to run from huggingface weights, but i am concerned
# about EMA etc. so think this is outside of the scope of establishing the test suite
INITIAL_CKPT_PATH = "/network/scratch/t/tanc/ablation_models/prose_up_to_8aa_standard_v1.ckpt"


@pytest.fixture(scope="function")
def cfg_test_self_improve(tmp_path: Path) -> DictConfig:
    """
    Parameterized Hydra-composed config for each experiment file under configs/experiment/evaluation.
    Each test that takes `cfg_eval_param` will run once per config file.

    This fixture safely resets Hydra between runs and patches all paths.
    """
    rel_path = EXPERIMENT_CONFIG.relative_to(CONFIG_BASE).with_suffix("")
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
        cfg.initial_ckpt_path = INITIAL_CKPT_PATH
        cfg.trainer.num_sanity_val_steps = 0  # disable val sanity checks
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.model.sampling_config.num_proposal_samples = 32
        cfg.model.sampling_config.self_improve_improve_proposal_samples = 32
        cfg.data.batch_size = 32
        cfg.data.test_sequences = "AA"
        cfg.tags = ["pytest", "test_self_improvement"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.slow
def test_self_improve(cfg_test_self_improve):
    """
    Runs eval() for every experiment config discovered via the fixture.
    :param cfg_eval_param: The configuration for the evaluation.
    """

    metrics, _ = self_improve(cfg_test_self_improve)

    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"
