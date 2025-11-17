"""
Tests for the evaluation pipelines.
If this test collection passes, we know that:
1. The evaluation pipeline can be run end-to-end for all different dataset.
2. All the huggingface model weights can be correctly loaded and used for evaluation.
3. The neural_network code hasn't broken since uploading model weights.
NOTE: This test only considers SNIS for each dataset/model. SMC tests are in test_smc.py
A very loose threshold on median proposal energy is used to catch major issues.
"""

from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.eval import eval
from tests.helpers.utils import compose_config, extract_test_sequence, get_config_stem

MEDIAN_PROPOSAL_ENERGY_THRESHOLDS = {  # these are intentionally loose, just to catch major issues
    "Ace-A-Nme": -10,
    "AAA": -120,
    "Ace-AAA-Nme": 50,
    "AAAAAA": -60,
    "GYDPETGTWG": -100,
    "AA": -160,
}

EXPERIMENT_NAMES = [
    f"evaluation/{cfg_path}"
    for cfg_path in [
        "single_system/ecnf++_Ace-A-Nme_snis.yaml",
        "single_system/ecnf++_AAA_snis.yaml",
        "single_system/ecnf++_Ace-AAA-Nme_snis.yaml",
        "single_system/ecnf++_AAAAAA_snis.yaml",
        "single_system/tarflow_Ace-A-Nme_ula.yaml",
        "single_system/tarflow_AAA_ula.yaml",
        "single_system/tarflow_Ace-AAA-Nme_ula.yaml",
        "single_system/tarflow_AAAAAA_ula.yaml",
        "single_system/tarflow_GYDPETGTWG_ula.yaml",
        "transferable/ecnf++_up_to_4aa_snis.yaml",
        "transferable/tarflow_up_to_8aa_snis.yaml",
        "transferable/prose_up_to_8aa_snis.yaml",
    ]
]


@pytest.fixture(params=EXPERIMENT_NAMES, ids=get_config_stem)
def cfg_test_snis_mwe(request: pytest.FixtureRequest, trainer_name_param: str, tmp_path: Path) -> DictConfig:
    """
    Hydra-composed config for the evaluation experiments.

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
        config_name="eval",
        overrides=[f"experiment={experiment_name}", f"trainer={trainer_name_param}"],
    )

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = Path.cwd()
        cfg.model.sampling_config.num_test_proposal_samples = 25
        if "ula" in experiment_name:
            # we disable SMC here for testing - we are mostly concerned with weights being correctly setup
            if cfg.model.get("smc_sampler") is not None:
                cfg.model.smc_sampler.enabled = False
        if "transferable" in experiment_name:
            cfg.data.test_sequences = "AA"
        cfg.data.num_workers = 0  # avoid multiprocessing issues in tests
        cfg.tags = ["pytest", f"test_snis_mwe_{trainer_name_param}"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM issues
@pytest.mark.pipeline
def test_snis_mwe(cfg_test_snis_mwe):
    """
    Run eval() for every experiment config provided by the `cfg_test_snis_mwe` fixture.

    Asserts:
    - 'test/{sequence}/proposal/median_energy' is present and below threshold.
    """

    metrics, _ = eval(cfg_test_snis_mwe)

    test_sequence = extract_test_sequence(cfg_test_snis_mwe)

    median_proposal_energy = metrics.get(f"test/{test_sequence}/proposal/median_energy", None)
    assert median_proposal_energy is not None, f"test/{test_sequence}/proposal/median_energy missing"
    assert median_proposal_energy < MEDIAN_PROPOSAL_ENERGY_THRESHOLDS[test_sequence], (
        f"Median proposal energy {median_proposal_energy} above threshold "
        f"{MEDIAN_PROPOSAL_ENERGY_THRESHOLDS[test_sequence]} for sequence {test_sequence}"
    )
