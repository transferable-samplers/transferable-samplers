"""
Tests for the self-improvement pipeline.
NOTE: Currently there is a hard-coded path to a checkpoint in this test, which needs to be
updated to point to a valid checkpoint on your system.
The entire self-improvement pipeline needs to be reworked to run from huggingface weights,
but this is outside the scope of establishing the test suite.
NOTE: A very basic test that only checks a single iteration and non-NaN loss.
"""

import os
from math import isnan
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.self_improve import self_improve
from tests.helpers.utils import compose_config, get_config_stem

EXPERIMENT_NAMES = ["evaluation/transferable/prose_up_to_8aa_self_improve.yaml"]
# TODO we need to setup self_improve to run from huggingface weights, but i am concerned
# about EMA etc. so think this is outside of the scope of establishing the test suite
INITIAL_CKPT_PATH = "/network/scratch/t/tanc/ablation_models/prose_up_to_8aa_standard_v1.ckpt"


@pytest.fixture(params=EXPERIMENT_NAMES, ids=lambda p: get_config_stem(p), scope="function")
def cfg_test_self_improve_mwe(request: pytest.FixtureRequest, trainer_name_param: str, tmp_path: Path) -> DictConfig:
    """
    Hydra-composed config for the transferable self-improvement experiment.
    NOTE: Currently only a single config is used for this test, can be extended later.

    Args:
        tmp_path: pytest-provided temporary directory path.
        trainer_name_param: trainer name parameter supplied by parametrization fixtures.

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
        cfg.initial_ckpt_path = INITIAL_CKPT_PATH
        cfg.trainer.num_sanity_val_steps = 0  # disable val sanity checks
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.model.sampling_config.num_proposal_samples = 32
        cfg.model.sampling_config.num_self_improve_proposal_samples = 32
        cfg.data.batch_size = 32
        cfg.data.test_sequences = "AA"
        cfg.tags = ["pytest", f"test_self_improve_mwe_{trainer_name_param}"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM issues
@pytest.mark.pipeline
def test_self_improve_mwe(cfg_test_self_improve_mwe: DictConfig) -> None:
    """
    Run the self-improvement pipeline for a single iteration and check basic metrics.

    Args:
        cfg_test_self_improve_mwe: The composed DictConfig produced by the fixture.

    Asserts:
        - 'train/loss' is present in returned metrics and is not NaN.
    """
    metrics, _ = self_improve(cfg_test_self_improve_mwe)

    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"
