"""
Tests for the SMC pipelines.
Only a basic test to ensure SMC runs end-to-end for single-system / transferable models.
NOTE: A very loose threshold on median SMC energy is used to catch major issues.
"""

import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.eval import eval

MEDIAN_SMC_ENERGY_THRESHOLDS = {  # these are intentionally loose, just to catch major issues
    "AAA": -120,
    "AA": -160,
}

# Locate relevant experiment config files
CONFIG_BASE = Path(__file__).resolve().parent.parent / "configs"
EVAL_DIR = CONFIG_BASE / "experiment" / "evaluation"
EXPERIMENT_CONFIGS = [
    EVAL_DIR / Path(p)
    for p in [
        "single_system/tarflow_AAA_ula.yaml",
        "transferable/prose_up_to_8aa_mala.yaml",
    ]
]


@pytest.fixture(params=EXPERIMENT_CONFIGS, ids=lambda p: p.stem, scope="function")
def cfg_test_smc(request: pytest.FixtureRequest, trainer_name_param: str, tmp_path: Path) -> DictConfig:
    """
    Hydra-composed config for the evaluation experiments.

    Args:
        request: pytest request object to get the experiment override parameter.
        trainer_name_param: trainer name parameter supplied by parametrization fixtures.
            (currently unused, need to impelement DDP SMC)
        tmp_path: pytest-provided temporary directory path.

    Returns:
        DictConfig: Composed and patched Hydra config for the test.
    """
    rel_path = request.param.relative_to(CONFIG_BASE).with_suffix("")
    override = rel_path.as_posix().removeprefix("experiment/")

    # Important: clear Hydra before initializing
    GlobalHydra.instance().clear()

    # Compose full Hydra config
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval", overrides=[f"experiment={override}"])

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = os.getcwd()
        cfg.model.sampling_config.num_test_proposal_samples = 25
        cfg.model.sampling_config.num_smc_samples = 25
        cfg.model.smc_sampler.num_timesteps = 10
        if "transferable" in override:
            cfg.data.test_sequences = "AA"
        cfg.tags = ["pytest", "test_smc"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.skipif(torch.cuda.device_count() > 1, reason="Not yet implemented for DDP")
def test_smc(cfg_test_smc: DictConfig) -> None:
    """
    Run eval() for every experiment config provided by the `cfg_test_smc` fixture.

    Asserts:
    - 'test/{sequence}/smc/median_energy' is present and below threshold.
    """
    metrics, _ = eval(cfg_test_smc)

    if "sequence" in cfg_test_smc.data:
        test_sequence = cfg_test_smc.data.sequence
    else:
        test_sequence = cfg_test_smc.data.test_sequences
        if isinstance(test_sequence, list):
            test_sequence = test_sequence[0]
        assert test_sequence == "AA", "Only 'AA' sequence is expected in tests."

    median_smc_energy = metrics.get(f"test/{test_sequence}/smc/median_energy", None)
    assert median_smc_energy is not None, f"test/{test_sequence}/smc/median_energy missing"
    assert median_smc_energy < MEDIAN_SMC_ENERGY_THRESHOLDS[test_sequence], (
        f"Median smc energy {median_smc_energy} above threshold "
        f"{MEDIAN_SMC_ENERGY_THRESHOLDS[test_sequence]} for sequence {test_sequence}"
    )
