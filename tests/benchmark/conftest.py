"""
Benchmark-specific conftest.
- ``benchmark`` tests require GPU (skip on CPU).
- ``benchmark_ddp`` tests require DDP (skip on CPU and single-GPU).
"""

import pytest


@pytest.fixture(autouse=True)
def _enforce_benchmark_hardware(request, trainer_name_param: str):
    if request.node.get_closest_marker("benchmark_ddp"):
        if trainer_name_param not in ("ddp", "ddp_fork"):
            pytest.skip("benchmark_ddp tests require DDP")
    elif request.node.get_closest_marker("benchmark"):
        if trainer_name_param == "cpu":
            pytest.skip("Benchmark tests require GPU or DDP")
