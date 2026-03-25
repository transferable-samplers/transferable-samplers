"""
Tests that all experiment configs compose correctly via Hydra.

Validates:
- Composed configs match reference JSON snapshots (catches any config drift)
- All _target_ paths are importable

To update reference files after an intentional config change:
    python -m pytest tests/configs/test_experiment_configs.py --update-reference
"""

import importlib.util
import json
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from tests.helpers.utils import compose_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
EXPERIMENT_DIR = CONFIGS_DIR / "experiment"
REFERENCE_DIR = Path(__file__).resolve().parent / "reference_configs"

# Auto-discover all experiment YAML files
TRAINING_EXPERIMENTS = sorted(str(p.relative_to(EXPERIMENT_DIR)) for p in EXPERIMENT_DIR.glob("*/train/**/*.yaml"))
FINETUNE_EXPERIMENTS = sorted(str(p.relative_to(EXPERIMENT_DIR)) for p in EXPERIMENT_DIR.glob("*/finetune/**/*.yaml"))
EVALUATION_EXPERIMENTS = sorted(str(p.relative_to(EXPERIMENT_DIR)) for p in EXPERIMENT_DIR.glob("*/eval/**/*.yaml"))
ALL_EXPERIMENTS = (
    [("train", exp) for exp in TRAINING_EXPERIMENTS]
    + [("train", exp) for exp in FINETUNE_EXPERIMENTS]  # finetune uses the train entrypoint
    + [("eval", exp) for exp in EVALUATION_EXPERIMENTS]
)
ALL_EXPERIMENT_IDS = [Path(exp).stem for _, exp in ALL_EXPERIMENTS]


def _compose(config_name: str, experiment_path: str) -> dict:
    """Compose an experiment config and return as a plain dict (unresolved)."""
    cfg = compose_config(
        config_name=config_name,
        overrides=[f"experiment={experiment_path}"],
    )
    return OmegaConf.to_container(cfg, resolve=False)


def _resolve_target(target: str) -> bool:
    """Check that a dotted _target_ path points to a findable module."""
    parts = target.rsplit(".", 1)
    if len(parts) != 2:
        return False
    module_path, _ = parts
    return importlib.util.find_spec(module_path) is not None


def _collect_targets(cfg: DictConfig) -> list[str]:
    """Recursively collect all _target_ values from a DictConfig (no resolution needed)."""
    targets = []
    for key in cfg:
        try:
            val = cfg[key]
        except Exception:  # noqa: S112
            continue
        if key == "_target_":
            targets.append(str(val))
        elif OmegaConf.is_dict(val):
            targets.extend(_collect_targets(val))
    return targets


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Clear Hydra global state before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.mark.essential
@pytest.mark.parametrize(("config_name", "experiment_path"), ALL_EXPERIMENTS, ids=ALL_EXPERIMENT_IDS)
def test_experiment_config_drift(config_name: str, experiment_path: str, request) -> None:
    """Test that the composed config matches the reference JSON snapshot."""
    update = request.config.getoption("--update-reference")
    stem = Path(experiment_path).stem
    reference_path = REFERENCE_DIR / f"{stem}.json"

    composed = _compose(config_name, experiment_path)

    if update:
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        with reference_path.open("w") as f:
            json.dump(composed, f, indent=2, default=str, sort_keys=True)
            f.write("\n")
        pytest.skip(f"Updated reference file: {reference_path.name}")

    assert reference_path.exists(), (
        f"Reference file missing: {reference_path}\nRun with --update-reference to generate it."
    )

    with reference_path.open() as f:
        expected = json.load(f)

    actual_json = json.dumps(composed, indent=2, default=str, sort_keys=True)
    expected_json = json.dumps(expected, indent=2, default=str, sort_keys=True)

    assert actual_json == expected_json, (
        f"Config drift detected for {stem}.\nRun with --update-reference to accept the new config."
    )


@pytest.mark.essential
@pytest.mark.parametrize(("config_name", "experiment_path"), ALL_EXPERIMENTS, ids=ALL_EXPERIMENT_IDS)
def test_experiment_targets_importable(config_name: str, experiment_path: str) -> None:
    """Test that all _target_ paths in the config are importable."""
    cfg = compose_config(
        config_name=config_name,
        overrides=[f"experiment={experiment_path}"],
    )

    targets = _collect_targets(cfg)
    assert len(targets) > 0, "No _target_ found in config — likely misconfigured"

    for target in targets:
        assert _resolve_target(target), f"Cannot import _target_: {target}"
