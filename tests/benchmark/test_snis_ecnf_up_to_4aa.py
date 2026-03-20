"""Benchmark: SNIS evaluation for transferable ECNF++ on up_to_4aa.

Resampled metrics on AA. Refactor values: 5 seeds (42, 123, 456, 789, 1024), 20th March 2026.

Metric     | Paper*  | Refactor (mean ± std)
---------- | ------- | ----------------------
ESS        | 0.0515  | 0.0755  ± 0.0065
energy-w2  | 1.396   | 1.061   ± 0.229
torus-w2   | 0.204   | 0.173   ± 0.047
tica-w2    | 0.00260 | 0.00210 ± 0.00030

*Paper values are those used in the 2AA mean values in
Table 2 of Amortized Sampling with Transferable Normalizing Flows
https://arxiv.org/abs/2508.18175

Comments: the refactored values are within expected variability.
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
    "resampled/effective-sample-size": (0.0755, 0.0065),
    "resampled/energy-w2": (1.061, 0.229),
    "resampled/torus-w2": (0.173, 0.047),
    "resampled/tica-w2": (0.0021, 0.0003),
}


@pytest.mark.benchmark
def test_snis_ecnf_up_to_4aa(trainer_name_param: str, tmp_path: Path) -> None:
    GlobalHydra.instance().clear()
    cfg = compose_config(
        config_name="eval",
        overrides=["experiment=transferable/eval/ecnf++_up_to_4aa_snis.yaml", f"trainer={trainer_name_param}"],
    )
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.callbacks.sampling_evaluation.sampler.num_samples = 10_000
        cfg.data.test_sequences = "AA"
        # AA is small (~22 atoms); default batch sizes (6-8) are tuned for 4aa (~70 atoms)
        cfg.model.source_energy_config.sample_batch_size = 128
        cfg.model.source_energy_config.energy_batch_size = 128
        cfg.model.source_energy_config.grad_batch_size = 128
        cfg.seed = int(os.environ.get("PYTEST_SEED", 42))
        cfg.tags = ["pytest", "benchmark_snis"]

    metrics, _ = eval(cfg)
    GlobalHydra.instance().clear()

    test_sequence = extract_test_sequence(cfg)
    assert f"test/{test_sequence}/proposal/median-energy" in metrics

    for suffix, (ref_mean, ref_std) in REFERENCE.items():
        key = f"test/{test_sequence}/{suffix}"
        assert key in metrics, f"{key} missing from metrics"
        val = float(metrics[key])
        lo, hi = ref_mean - 2.5 * ref_std, ref_mean + 2.5 * ref_std
        assert lo <= val <= hi, (
            f"{key}={val:.4f} outside 2.5σ range [{lo:.4f}, {hi:.4f}] (ref {ref_mean:.4f}±{ref_std:.4f})"
        )

    print("\n--- Benchmark metrics ---")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")
