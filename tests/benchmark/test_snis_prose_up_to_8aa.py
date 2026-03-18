"""Benchmark: SNIS evaluation for transferable Prose on up_to_8aa.

Reference values on ARIP (10k samples, 3 seeds: 42, 123, 456):
    ESS:       0.0226 ± 0.0013
    energy-w2: 1.413  ± 0.411
    torus-w2:  0.628  ± 0.052
    tica-w2:   0.279  ± 0.082

Paper reference values on ARIP.
    Resampled:
        ESS:       0.0285
        energy-w2: 2.129
        tica-w2:   0.317
        torus-w2:  0.575
    Proposal:
        energy-w2: 552396672
        tica-w2:   0.688
        torus-w2:  1.986
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
    "resampled/effective-sample-size": (0.0226, 0.0013),
    "resampled/energy-w2": (1.413, 0.411),
    "resampled/torus-w2": (0.628, 0.052),
    "resampled/tica-w2": (0.279, 0.082),
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
    """Check metrics are within 2σ of reference values."""
    seed = int(os.environ.get("PYTEST_SEED", 42))
    cfg = _make_cfg(trainer_name_param, tmp_path, seed=seed)
    metrics, _ = eval(cfg)
    GlobalHydra.instance().clear()

    for suffix, (ref_mean, ref_std) in REFERENCE.items():
        key = f"test/ARIP/{suffix}"
        assert key in metrics, f"{key} missing from metrics"
        val = float(metrics[key])
        lo, hi = ref_mean - 2 * ref_std, ref_mean + 2 * ref_std
        assert lo <= val <= hi, (
            f"{key}={val:.4f} outside 2σ range [{lo:.4f}, {hi:.4f}] (ref {ref_mean:.4f}±{ref_std:.4f})"
        )
