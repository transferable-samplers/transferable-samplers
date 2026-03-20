"""Benchmark: Self-improvement pipeline for transferable Prose on up_to_8aa.

Resampled metrics on ARIP. Refactor values: 5 seeds (42, 123, 456, 789, 1024), 20th March 2026.
Commit: refactor@d85ecf16d76c97040d03f66c651a8f55b63cf671

Metric     | Paper* | Refactor (mean ± std)
---------- | ------ | ----------------------
ESS        | 0.024  | 0.0246 ± ~0
energy-w2  | 1.634  | 1.821  ± 0.089
torus-w2   | 0.494  | 0.522  ± 0.013
tica-w2    | 0.254  | 0.268  ± 0.020

*Paper values are those used in the 4AA mean values in
Table 4 of Amortized Sampling with Transferable Normalizing Flows
https://arxiv.org/abs/2508.18175

Comments: the refactored values are within expected variability.
"""

import os
from math import isnan
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config
from transferable_samplers.train import train

REFERENCE = {
    "resampled/energy-w2": (1.821, 0.089),
    "resampled/torus-w2": (0.522, 0.013),
    "resampled/tica-w2": (0.268, 0.020),
}


@pytest.mark.benchmark
def test_self_improve_prose_up_to_8aa(trainer_name_param: str, tmp_path: Path) -> None:
    GlobalHydra.instance().clear()
    cfg = compose_config(
        config_name="train",
        overrides=[
            "experiment=transferable/fine-tune/prose_up_to_8aa_self_improve.yaml",
            f"trainer={trainer_name_param}",
        ],
    )
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.data.test_sequences = "ARIP"
        cfg.seed = int(os.environ.get("PYTEST_SEED", 42))
        cfg.tags = ["pytest", "benchmark_self_improve"]

    metrics, _ = train(cfg)
    GlobalHydra.instance().clear()

    assert "train/loss" in metrics, "train/loss missing from metrics"
    assert not isnan(metrics["train/loss"]), "train/loss is NaN"

    for suffix, (ref_mean, ref_std) in REFERENCE.items():
        key = f"self-improve/ARIP/{suffix}"
        assert key in metrics, f"{key} missing from metrics"
        val = float(metrics[key])
        lo, hi = ref_mean - 2.5 * ref_std, ref_mean + 2.5 * ref_std
        assert lo <= val <= hi, (
            f"{key}={val:.4f} outside 2.5σ range [{lo:.4f}, {hi:.4f}] (ref {ref_mean:.4f}±{ref_std:.4f})"
        )

    print("\n--- Benchmark metrics ---")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")
