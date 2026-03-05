"""Benchmark: SNIS evaluation for transferable ECNF++ on up_to_4aa."""

from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

# pyrefly: ignore [missing-import]
from tests.helpers.utils import compose_config, extract_test_sequence
from transferable_samplers.eval import eval


@pytest.mark.forked
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
        cfg.tags = ["pytest", "benchmark_snis"]

    metrics, _ = eval(cfg)
    GlobalHydra.instance().clear()

    test_sequence = extract_test_sequence(cfg)
    assert f"test/{test_sequence}/proposal/median_energy" in metrics

    print("\n--- Benchmark metrics ---")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")
