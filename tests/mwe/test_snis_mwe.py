"""
Tests for the SNIS evaluation pipeline.
One tarflow + one ecnf config for each of single_system and transferable.
A very loose threshold on median proposal energy is used to catch major issues.
"""

from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config, extract_test_sequence, get_config_stem
from transferable_samplers.eval import eval

MEDIAN_PROPOSAL_ENERGY_THRESHOLDS = {  # these are intentionally loose, just to catch major issues
    "AAA": -120,
    "AA": -160,
}

EXPERIMENT_NAMES = [
    "single_system/eval/ecnf++_AAA_snis.yaml",  # single system ecnf
    "single_system/eval/tarflow_AAA_ula.yaml",  # single system tarflow
    "transferable/eval/ecnf++_up_to_4aa_snis.yaml",  # transferable ecnf
    "transferable/eval/prose_up_to_8aa_snis.yaml",  # transferable tarflow
]


@pytest.fixture(params=EXPERIMENT_NAMES, ids=get_config_stem)
# pyrefly: ignore [bad-return]
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
        config_name="eval", overrides=[f"experiment={experiment_name}", f"trainer={trainer_name_param}"]
    )

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.callbacks.sampling_evaluation.sampler.num_samples = 25
        if trainer_name_param == "cpu":
            cfg.callbacks.sampling_evaluation.run_diagnostics_kwargs = {
                "num_samples_invert": 8,
                "num_samples_dlogp": 2,
            }
        if "ula" in experiment_name or "mala" in experiment_name:
            # Replace SMC sampler with SNIS for this test.
            cfg.callbacks.sampling_evaluation.sampler = OmegaConf.create(
                {
                    "_target_": "transferable_samplers.samplers.snis_sampler.SNISSampler",
                    "num_samples": 25,
                }
            )
        if "transferable" in experiment_name:
            cfg.data.test_sequences = "AA"
        cfg.data.num_workers = 0  # avoid multiprocessing issues in tests
        cfg.data.batch_size = 4
        cfg.tags = ["pytest", f"test_snis_mwe_{trainer_name_param}"]

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.essential
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
