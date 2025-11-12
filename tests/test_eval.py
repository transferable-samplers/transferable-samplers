"""
If this test collection passes, we know that:
1. the evaluation pipeline can be run end-to-end for all different dataset
2. all the huggingface model weights can be correctly loaded and used for evaluation
"""

import os
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.eval import eval

MEDIAN_PROPOSAL_ENERGY_THRESHOLDS = {  # these are intentionally loose, just to catch major issues
    "Ace-A-Nme": -10,
    "AAA": -120,
    "Ace-AAA-Nme": 50,
    "AAAAAA": -80,
    "GYDPETGTWG": -400,
    "AA": -160,
}

# Locate relevant experiment config files
CONFIG_BASE = Path(__file__).resolve().parent.parent / "configs"
EVAL_DIR = CONFIG_BASE / "experiment" / "evaluation"
EXPERIMENT_CONFIGS = [
    EVAL_DIR / Path(p)
    for p in [
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


@pytest.fixture(params=EXPERIMENT_CONFIGS, ids=lambda p: p.stem, scope="function")
def cfg_test_evaluation_fast(request: pytest.FixtureRequest, tmp_path: Path) -> DictConfig:
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
        cfg = compose(config_name="eval", overrides=[f"experiment={override}"])

    # without "turning-off" the metrics (should be made available as an option),
    # this is the smallest value that will pass through the eval pipeline without error
    # TODO address this
    cfg.model.sampling_config.num_test_proposal_samples = 25
    if "ula" in override:
        # we disable SMC here for testing - we are mostly concerned with weights being correctly setup
        if cfg.model.get("smc_sampler") is not None:
            cfg.model.smc_sampler.enabled = False
    elif "transferable" in override:
        cfg.data.test_sequences = "AA"
    cfg.tags = ["pytest", "test_evaluation_fast"]

    # Patch common paths to avoid writing to the project tree
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = os.getcwd()

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.slow
def test_evaluation_fast(cfg_test_evaluation_fast):
    """
    Runs eval() for every experiment config discovered via the fixture.
    :param cfg_eval_param: The configuration for the evaluation.
    """

    metrics, _ = eval(cfg_test_evaluation_fast)

    if "sequence" in cfg_test_evaluation_fast.data:
        test_sequence = cfg_test_evaluation_fast.data.sequence
    else:
        test_sequence = cfg_test_evaluation_fast.data.test_sequences
        if isinstance(test_sequence, list):
            test_sequence = test_sequence[0]
        assert test_sequence == "AA", "Only 'AA' sequence is expected in tests."

    median_proposal_energy = metrics.get(f"test/{test_sequence}/proposal/median_energy", None)
    assert median_proposal_energy is not None, f"test/{test_sequence}/proposal/median_energy missing"

    with open("output", "a", encoding="utf-8") as f:
        f.write(f"{test_sequence}\t{cfg_test_evaluation_fast.state_dict_hf_path}\t{median_proposal_energy}\n")

    assert median_proposal_energy < MEDIAN_PROPOSAL_ENERGY_THRESHOLDS[test_sequence], (
        f"Median proposal energy {median_proposal_energy} above threshold "
        f"{MEDIAN_PROPOSAL_ENERGY_THRESHOLDS[test_sequence]} for sequence {test_sequence}"
    )
