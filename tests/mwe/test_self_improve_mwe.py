"""
Tests for the self-improvement pipeline.
NOTE: A very basic test that only checks a single iteration and non-NaN loss.
"""

from math import isnan
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config
from transferable_samplers.train import train

# to keep the network small use this experiment config and override to add self-improve
EXPERIMENT = "single_system/train/tarflow_AAA.yaml"


@pytest.fixture
# pyrefly: ignore [bad-return]
def cfg_test_self_improve_mwe(trainer_name_param: str, tmp_path: Path) -> DictConfig:
    """
    Hydra-composed config for the single-system TarFlow self-improvement experiment.
    Builds on the tarflow_AAA train config, adding self-improve overrides inline.

    Args:
        tmp_path: pytest-provided temporary directory path.
        trainer_name_param: trainer name parameter supplied by parametrization fixtures.

    Returns:
        DictConfig: Composed and patched Hydra config for the test.
    """
    # Important: clear Hydra before initializing
    GlobalHydra.instance().clear()

    cfg = compose_config(config_name="train", overrides=[f"experiment={EXPERIMENT}", f"trainer={trainer_name_param}"])

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.trainer.num_sanity_val_steps = 0  # disable val sanity checks
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.data.num_workers = 0  # avoid multiprocessing issues in tests
        cfg.data.batch_size = 2
        cfg.data.train_from_buffer = True
        # Shrink network for faster CI runs
        cfg.model.net.channels = 64
        cfg.model.net.num_blocks = 2
        cfg.model.net.layers_per_block = 2
        # Self-improve model settings
        cfg.model.train_from_buffer = True
        cfg.model.teacher_regularize_weight = 0.5
        # Disable EMA: incompatible with PopulateBufferCallback
        cfg.callbacks.ema = None
        # Add populate_buffer callback with a small sample count for testing
        cfg.callbacks.populate_buffer = OmegaConf.create(
            {
                "_target_": "transferable_samplers.callbacks.populate_buffer_callback.PopulateBufferCallback",
                "sampler": {
                    "_target_": "transferable_samplers.samplers.snis_sampler.SNISSampler",
                    "num_samples": 32,
                    "logw_quantile_filter": 0.002,
                },
            }
        )
        if trainer_name_param == "cpu":
            cfg.callbacks.sampling_evaluation.run_diagnostics_kwargs = {
                "num_samples_invert": 8,
                "num_samples_dlogp": 2,
            }
        cfg.tags = ["pytest", f"test_self_improve_mwe_{trainer_name_param}"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


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
