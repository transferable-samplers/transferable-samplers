"""
Tests for the SMC pipelines.
Only a basic test to ensure SMC runs end-to-end for single-system / transferable models.
NOTE: A very loose threshold on median SMC energy is used to catch major issues.
"""

import os
from pathlib import Path

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.eval import eval
from tests.helpers.utils import compose_config, extract_test_sequence, get_config_stem

MEDIAN_SMC_ENERGY_THRESHOLDS = {  # these are intentionally loose, just to catch major issues
    "AAA": -120,
    "AA": -160,
}

EXPERIMENT_NAMES = [
    f"evaluation/{cfg_path}"
    for cfg_path in [
        "single_system/tarflow_AAA_ula.yaml",
        "transferable/prose_up_to_8aa_mala.yaml",
    ]
]


@pytest.fixture(params=EXPERIMENT_NAMES, ids=get_config_stem, scope="function")
def cfg_test_smc_mwe(request: pytest.FixtureRequest, trainer_name_param: str, tmp_path: Path) -> DictConfig:
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
    # Important: clear Hydra before initializing
    GlobalHydra.instance().clear()

    experiment_name = request.param

    cfg = compose_config(
        config_name="eval", overrides=[f"experiment={experiment_name}", f"trainer={trainer_name_param}"]
    )

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = os.getcwd()
        cfg.model.proposal_config.num_test_proposal_samples = 25
        cfg.model.proposal_config.num_smc_samples = 25
        cfg.model.smc_sampler.num_timesteps = 10
        cfg.data.num_workers = 0  # avoid multiprocessing issues in tests
        if "transferable" in experiment_name:
            cfg.data.test_sequences = "AA"
        cfg.tags = ["pytest", "test_smc_mwe"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM issues
@pytest.mark.pipeline
@pytest.mark.skipif(torch.cuda.device_count() > 1, reason="Not yet implemented for DDP")
def test_smc_mwe(cfg_test_smc_mwe: DictConfig) -> None:
    """
    Run eval() for every experiment config provided by the `cfg_test_smc_mwe` fixture.

    Asserts:
    - 'test/{sequence}/smc/median_energy' is present and below threshold.
    """
    metrics, _ = eval(cfg_test_smc_mwe)

    test_sequence = extract_test_sequence(cfg_test_smc_mwe)

    median_smc_energy = metrics.get(f"test/{test_sequence}/smc/median_energy", None)
    assert median_smc_energy is not None, f"test/{test_sequence}/smc/median_energy missing"
    assert median_smc_energy < MEDIAN_SMC_ENERGY_THRESHOLDS[test_sequence], (
        f"Median smc energy {median_smc_energy} above threshold "
        f"{MEDIAN_SMC_ENERGY_THRESHOLDS[test_sequence]} for sequence {test_sequence}"
    )
