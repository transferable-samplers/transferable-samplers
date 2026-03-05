"""Benchmark: Self-improvement pipeline for transferable Prose on up_to_8aa."""

from math import isnan
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config
from transferable_samplers.train import train


@pytest.mark.benchmark
def test_self_improve_prose_up_to_8aa(trainer_name_param: str, tmp_path: Path) -> None:
    GlobalHydra.instance().clear()
    cfg = compose_config(
        config_name="train",
        overrides=[
            "experiment=transferable/finetune/prose_up_to_8aa_self_improve.yaml",
            f"trainer={trainer_name_param}",
        ],
    )
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.data.test_sequences = "ARIP"
        cfg.tags = ["pytest", "benchmark_self_improve"]

    metrics, _ = train(cfg)
    GlobalHydra.instance().clear()

    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"

    print("\n--- Benchmark metrics ---")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")
