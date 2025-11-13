"""
Fixtures for configuring tests based on available hardware (GPU/DPD).
Tests are run with the appropriate trainer based on available devices.
Unique ids for each trainer are provided for clarity in test reports.
"""

import os

import pytest
import torch

# Create report directory if it doesn't exist
report_dir = os.environ.get("PYTEST_REPORT_DIR", "tests/")
os.makedirs(report_dir, exist_ok=True)


# Tests use this fixture to parametrize over trainer names will now run with
# correct trainer, with unique ids for clarity in test reports.
@pytest.fixture(scope="session")
def trainer_name_param():
    trainer = os.environ.get("PYTEST_TRAINER", "gpu")
    if trainer == "ddp" and torch.cuda.device_count() < 2:
        pytest.skip("DDP requires >=2 GPUs")
    if trainer == "gpu" and torch.cuda.device_count() < 1:
        pytest.skip("No GPU available")
    return trainer


def pytest_configure(config):
    config.addinivalue_line("markers", "pipeline: long-running pipeline tests")
    config.addinivalue_line("markers", "forked: force forked execution via pytest-forked")
