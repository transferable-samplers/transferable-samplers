"""
Benchmark tests for the SMC evaluation pipeline.
Runs full experiment configs and collates all returned metrics.
"""

from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config, extract_test_sequence, get_config_stem
from transferable_samplers.eval import eval

EXPERIMENT_NAMES = [
    "single_system/eval/tarflow_AAA_ula.yaml",
    "transferable/eval/prose_up_to_8aa_mala.yaml",
]


@pytest.fixture(params=EXPERIMENT_NAMES, ids=get_config_stem)
# pyrefly: ignore [bad-return]
def cfg_smc_benchmark(request: pytest.FixtureRequest, trainer_name_param: str, tmp_path: Path) -> DictConfig:
    GlobalHydra.instance().clear()

    experiment_name = request.param
    cfg = compose_config(
        config_name="eval", overrides=[f"experiment={experiment_name}", f"trainer={trainer_name_param}"]
    )

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        if "transferable" in experiment_name:
            cfg.data.test_sequences = "ARIP"
        cfg.tags = ["pytest", f"benchmark_smc_{trainer_name_param}"]

    yield cfg

    GlobalHydra.instance().clear()


def _run_smc_benchmark(cfg: DictConfig) -> None:
    # pyrefly: ignore [bad-argument-type]
    metrics, _ = eval(cfg)

    test_sequence = extract_test_sequence(cfg)

    median_key = f"test/{test_sequence}/smc/median_energy"
    assert median_key in metrics, f"{median_key} missing from metrics"

    print("\n--- Benchmark metrics ---")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")


@pytest.mark.forked
@pytest.mark.benchmark
def test_smc_benchmark(cfg_smc_benchmark: DictConfig) -> None:
    """Run full SMC evaluation experiment on GPU and collate all metrics."""
    _run_smc_benchmark(cfg_smc_benchmark)


@pytest.mark.forked
@pytest.mark.benchmark_ddp
def test_smc_benchmark_ddp(cfg_smc_benchmark: DictConfig) -> None:
    """Run full SMC evaluation experiment on DDP and collate all metrics."""
    _run_smc_benchmark(cfg_smc_benchmark)
