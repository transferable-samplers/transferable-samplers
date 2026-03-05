"""
Benchmark tests for the self-improvement pipeline.
Runs full experiment configs and collates all returned metrics.
"""

from math import isnan
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config, get_config_stem
from transferable_samplers.train import train

EXPERIMENT_NAMES = [
    "transferable/finetune/prose_up_to_8aa_self_improve.yaml",
]


@pytest.fixture(params=EXPERIMENT_NAMES, ids=get_config_stem)
# pyrefly: ignore [bad-return]
def cfg_self_improve_benchmark(request: pytest.FixtureRequest, trainer_name_param: str, tmp_path: Path) -> DictConfig:
    GlobalHydra.instance().clear()

    experiment_name = request.param
    cfg = compose_config(
        config_name="train", overrides=[f"experiment={experiment_name}", f"trainer={trainer_name_param}"]
    )

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.data.test_sequences = "ARIP"
        cfg.tags = ["pytest", f"benchmark_self_improve_{trainer_name_param}"]

    yield cfg

    GlobalHydra.instance().clear()


def _run_self_improve_benchmark(cfg: DictConfig) -> None:
    metrics, _ = train(cfg)

    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"

    print("\n--- Benchmark metrics ---")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")


@pytest.mark.forked
@pytest.mark.benchmark
def test_self_improve_benchmark(cfg_self_improve_benchmark: DictConfig) -> None:
    """Run full self-improvement experiment on GPU and collate all metrics."""
    _run_self_improve_benchmark(cfg_self_improve_benchmark)
