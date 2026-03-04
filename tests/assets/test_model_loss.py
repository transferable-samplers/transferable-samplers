"""
Tests that all model weights load correctly and produce a valid (finite) loss.
Uses the same eval pipeline as test_snis_mwe.py but checks eval_loss instead of proposal energy.
Runs on CPU with a single batch of 32 samples for deterministic reference values.
"""

from math import isnan
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config, extract_test_sequence, get_config_stem
from transferable_samplers.eval import eval

EXPERIMENT_NAMES = [
    "single_system/eval/ecnf++_Ace-A-Nme_snis.yaml",
    "single_system/eval/ecnf++_AAA_snis.yaml",
    "single_system/eval/ecnf++_Ace-AAA-Nme_snis.yaml",
    "single_system/eval/ecnf++_AAAAAA_snis.yaml",
    "single_system/eval/tarflow_Ace-A-Nme_ula.yaml",
    "single_system/eval/tarflow_AAA_ula.yaml",
    "single_system/eval/tarflow_Ace-AAA-Nme_ula.yaml",
    "single_system/eval/tarflow_AAAAAA_ula.yaml",
    "single_system/eval/tarflow_GYDPETGTWG_ula.yaml",
    "transferable/eval/ecnf++_up_to_4aa_snis.yaml",
    "transferable/eval/tarflow_up_to_8aa_snis.yaml",
    "transferable/eval/prose_up_to_8aa_snis.yaml",
]

# Reference loss values (CPU, batch_size=32, first 32 samples).
# Tests assert within 1% relative tolerance.
REFERENCE_LOSS = {
    "single_system/eval/ecnf++_Ace-A-Nme_snis.yaml": 38.5,
    "single_system/eval/ecnf++_AAA_snis.yaml": 52.5,
    "single_system/eval/ecnf++_Ace-AAA-Nme_snis.yaml": 53.5,
    "single_system/eval/ecnf++_AAAAAA_snis.yaml": 86.3,
    "single_system/eval/tarflow_Ace-A-Nme_ula.yaml": -98.9,
    "single_system/eval/tarflow_AAA_ula.yaml": -181.3,
    "single_system/eval/tarflow_Ace-AAA-Nme_ula.yaml": -246.8,
    "single_system/eval/tarflow_AAAAAA_ula.yaml": -425.4,
    "single_system/eval/tarflow_GYDPETGTWG_ula.yaml": -1033.9,
    "transferable/eval/ecnf++_up_to_4aa_snis.yaml": 22.1,
    "transferable/eval/tarflow_up_to_8aa_snis.yaml": -160.3,
    "transferable/eval/prose_up_to_8aa_snis.yaml": -160.4,
}


@pytest.fixture(params=EXPERIMENT_NAMES, ids=get_config_stem)
# pyrefly: ignore [bad-return]
def cfg_and_experiment_name(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[DictConfig, str]:
    """
    Hydra-composed config for eval experiments, with sampling disabled so only loss is computed.
    """
    GlobalHydra.instance().clear()

    experiment_name = request.param

    cfg = compose_config(config_name="eval", overrides=[f"experiment={experiment_name}", "trainer=cpu"])

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        # Disable sampling evaluation — we only care about loss
        cfg.callbacks.sampling_evaluation = None
        if "transferable" in experiment_name:
            cfg.data.test_sequences = "AA"
        cfg.data.num_workers = 0
        cfg.data.batch_size = 32
        cfg.callbacks.loss_evaluation.batch_size = 32
        cfg.callbacks.loss_evaluation.max_samples = 32
        cfg.seed = 42  # deterministic for reference values
        cfg.tags = ["pytest", "test_model_loss_cpu"]

    yield cfg, experiment_name

    GlobalHydra.instance().clear()


@pytest.mark.forked
@pytest.mark.essential
def test_model_loss(cfg_and_experiment_name: tuple[DictConfig, str]) -> None:
    """
    Run eval() for every experiment config and check that:
    1. The model weights load successfully (eval doesn't crash).
    2. The eval_loss metric is present and is finite (not NaN).
    3. The loss matches the reference value within 1% relative tolerance.
    """
    cfg, experiment_name = cfg_and_experiment_name
    metrics, _ = eval(cfg)

    test_sequence = extract_test_sequence(cfg)

    loss_key = f"test/{test_sequence}/eval_loss"
    assert loss_key in metrics, f"{loss_key} missing from metrics"

    loss = metrics[loss_key]
    print(f"\n{loss_key}: {loss:.1f}")
    assert not isnan(loss), f"{loss_key} is NaN"

    ref = REFERENCE_LOSS[experiment_name]
    assert abs(loss - ref) / abs(ref) < 0.01, (
        f"{loss_key} = {loss:.1f} differs from reference {ref:.1f} by more than 1%"
    )
