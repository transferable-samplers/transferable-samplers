"""Benchmark: SMC evaluation for transferable Prose on up_to_8aa with MALA.

SMC metrics on ARIP. Refactor values: 5 seeds (42, 123, 456, 789, 1024), 20th March 2026.
Commit: refactor@d85ecf16d76c97040d03f66c651a8f55b63cf671

Metric     | Paper* | Refactor (mean ± std)
---------- | ------ | ----------------------
energy-w2  | 3.563  | 1.282  ± 0.252
torus-w2   | 0.763  | 0.619  ± 0.082
tica-w2    | 0.459  | 0.308  ± 0.091

*Paper values are those used in the 4AA mean values in
Table 4 of Amortized Sampling with Transferable Normalizing Flows
https://arxiv.org/abs/2508.18175

Comments: I am investigating the increase in performance, it may be due to differing model weights employed.
"""

import os
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config, extract_test_sequence
from transferable_samplers.eval import eval

REFERENCE = {
    "smc/energy-w2": (1.282, 0.252),
    "smc/torus-w2": (0.619, 0.082),
    "smc/tica-w2": (0.308, 0.091),
}


@pytest.mark.benchmark
def test_smc_prose_up_to_8aa(trainer_name_param: str, tmp_path: Path) -> None:
    GlobalHydra.instance().clear()
    cfg = compose_config(
        config_name="eval",
        overrides=["experiment=transferable/eval/prose_up_to_8aa_mala.yaml", f"trainer={trainer_name_param}"],
    )
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.data.test_sequences = "ARIP"
        cfg.model.source_energy_config.sample_batch_size = 1_024
        cfg.model.source_energy_config.energy_batch_size = 128
        cfg.model.source_energy_config.grad_batch_size = 64
        cfg.seed = int(os.environ.get("PYTEST_SEED", 42))
        cfg.tags = ["pytest", "benchmark", "benchmark_smc_prose_up_to_8aa"]

    # pyrefly: ignore [bad-argument-type]
    metrics, _ = eval(cfg)
    GlobalHydra.instance().clear()

    test_sequence = extract_test_sequence(cfg)
    assert f"test/{test_sequence}/smc/median-energy" in metrics

    print("\n--- Benchmark metrics ---")
    for suffix in REFERENCE:
        key = f"test/{test_sequence}/{suffix}"
        print(f"  {key}: {metrics.get(key, 'MISSING')}")

    for suffix, (ref_mean, ref_std) in REFERENCE.items():
        key = f"test/{test_sequence}/{suffix}"
        assert key in metrics, f"{key} missing from metrics"
        val = float(metrics[key])
        lo, hi = ref_mean - 2.5 * ref_std, ref_mean + 2.5 * ref_std
        assert lo <= val <= hi, (
            f"{key}={val:.4f} outside 2.5σ range [{lo:.4f}, {hi:.4f}] (ref {ref_mean:.4f}±{ref_std:.4f})"
        )
