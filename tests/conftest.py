"""
Fixtures for configuring tests based on available hardware (GPU/DDP).
Tests are run with the appropriate trainer based on available devices.
Unique ids for each trainer are provided for clarity in test reports.
"""

import os
from pathlib import Path

import pytest
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

# Create report directory if it doesn't exist
report_dir = os.environ.get("PYTEST_REPORT_DIR", "tests/")
Path(report_dir).mkdir(parents=True, exist_ok=True)


# Used to inject trainer name parameter into tests
@pytest.fixture(scope="session", params=[os.environ.get("PYTEST_TRAINER", "gpu")], ids=lambda x: x)
def trainer_name_param(request):
    trainer = request.param

    if trainer in ("ddp", "ddp_fork") and torch.cuda.device_count() < 2:
        pytest.skip("DDP requires >=2 GPUs")
    if trainer == "gpu" and torch.cuda.device_count() < 1:
        pytest.skip("No GPU available")

    # ddp strategy requires torchrun/SLURM launcher; use ddp_fork in tests so
    # Lightning forks worker processes itself without an external launcher.
    if trainer == "ddp":
        return "ddp_fork"
    return trainer


def pytest_addoption(parser):
    parser.addoption(
        "--update-reference", action="store_true", default=False, help="Regenerate reference config JSON files"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "essential: must-pass tests (config, assets, algorithms)")
    config.addinivalue_line("markers", "benchmark: long-running runs that validate metrics against a paper")
    config.addinivalue_line("markers", "optional: architectural/design tests")


@pytest.fixture(autouse=True)
def _enforce_hardware_constraints(request, trainer_name_param: str):
    if request.node.get_closest_marker("benchmark"):
        if trainer_name_param == "cpu":
            pytest.skip("Benchmark tests require GPU or DDP")
    if (
        trainer_name_param == "ddp_fork"
        and request.node.get_closest_marker("essential")
        and "tests/mwe" not in str(request.fspath)
    ):
        pytest.skip("DDP essential tests only run MWE tests")
    if "tests/configs" in str(request.fspath) and trainer_name_param != "cpu":
        pytest.skip("Config tests only run on CPU")
