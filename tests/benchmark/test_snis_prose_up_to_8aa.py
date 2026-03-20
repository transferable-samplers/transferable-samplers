"""Benchmark: SNIS evaluation for transferable Prose on up_to_8aa.

Resampled metrics on ARIP. Refactor values: 5 seeds (42, 123, 456, 789, 1024), 20th March 2026.
Commit: refactor@d85ecf16d76c97040d03f66c651a8f55b63cf671

Metric     | Paper* | Refactor (mean ± std)
---------- | ------ | ----------------------
ESS        | 0.0285 | 0.0248 ± 0.0029
energy-w2  | 2.129  | 1.484  ± 0.369
torus-w2   | 0.575  | 0.635  ± 0.034
tica-w2    | 0.317  | 0.321  ± 0.073

*Paper values are those used in the 8AA sequence-wise means in
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
from tests.helpers.utils import compose_config
from transferable_samplers.eval import eval

REFERENCE = {
    "resampled/effective-sample-size": (0.0248, 0.0029),
    "resampled/energy-w2": (1.484, 0.369),
    "resampled/torus-w2": (0.635, 0.034),
    "resampled/tica-w2": (0.321, 0.073),
}


def _make_cfg(trainer_name_param: str, tmp_path: Path, seed: int = 42):
    GlobalHydra.instance().clear()
    cfg = compose_config(
        config_name="eval",
        overrides=["experiment=transferable/eval/prose_up_to_8aa_snis.yaml", f"trainer={trainer_name_param}"],
    )
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(Path.cwd())
        cfg.callbacks.sampling_evaluation.sampler.num_samples = 10_000
        cfg.data.test_sequences = "ARIP"
        cfg.seed = seed
        cfg.tags = ["pytest", "benchmark_snis"]
    return cfg


@pytest.mark.benchmark
def test_snis_prose_up_to_8aa(trainer_name_param: str, tmp_path: Path) -> None:
    """Check metrics are within 2.5σ of reference values."""
    seed = int(os.environ.get("PYTEST_SEED", 42))
    cfg = _make_cfg(trainer_name_param, tmp_path, seed=seed)
    metrics, _ = eval(cfg)
    GlobalHydra.instance().clear()

    for suffix, (ref_mean, ref_std) in REFERENCE.items():
        key = f"test/ARIP/{suffix}"
        assert key in metrics, f"{key} missing from metrics"
        val = float(metrics[key])
        lo, hi = ref_mean - 2.5 * ref_std, ref_mean + 2.5 * ref_std
        assert lo <= val <= hi, (
            f"{key}={val:.4f} outside 2.5σ range [{lo:.4f}, {hi:.4f}] (ref {ref_mean:.4f}±{ref_std:.4f})"
        )
