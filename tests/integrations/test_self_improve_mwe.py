"""
Tests for the self-improvement pipeline.
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

EXPERIMENT_NAMES = ["evaluation/transferable/prose_up_to_8aa_self_improve.yaml"]


@pytest.fixture(params=EXPERIMENT_NAMES, ids=get_config_stem, scope="function")
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
        cfg.trainer.num_sanity_val_steps = 0  # disable val sanity checks
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.callbacks.populate_buffer.sampler.num_samples = 16
        if trainer_name_param == "cpu":
            cfg.callbacks.sampling_evaluation.run_diagnostics_kwargs = {
                "num_samples_invert": 8, "num_samples_dlogp": 2,
            }
        cfg.data.num_workers = 0  # avoid multiprocessing issues in tests
        cfg.data.batch_size = 4
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
    metrics, _ = train(cfg_test_self_improve_mwe)

    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"
