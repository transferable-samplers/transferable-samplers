"""
Fixtures for configuring tests based on available hardware (GPU).
Tests are run with the appropriate trainer based on available devices.
Unique ids for each trainer are provided for clarity in test reports.
"""

import gc
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
@pytest.fixture(
    scope="session",
    params=[os.environ.get("PYTEST_TRAINER", "cpu")],  # default to "cpu" if not set
    ids=lambda x: x,
)
def trainer_name_param(request):
    trainer = request.param

    if trainer == "gpu" and torch.cuda.device_count() < 1:
        pytest.skip("No GPU available")

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
            pytest.skip("Benchmark tests require GPU")


@pytest.fixture(autouse=True)
def _free_memory_after_test():
    yield
    gc.collect()
    torch.cuda.empty_cache()
