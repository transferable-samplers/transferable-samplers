import os

import pytest
import torch

# Have to export env variable for DDP tests
trainer_name = os.environ.get("PYTEST_TRAINER", "gpu")  # default: gpu

# Ensure correct trainer is used based on available devices
num_devices = torch.cuda.device_count()
if num_devices == 0:
    raise RuntimeError("No GPU available for tests in test_train.py")
elif num_devices == 1:
    assert trainer_name == "gpu", "Only single GPU available, cannot run DDP tests"
else:
    assert trainer_name == "ddp", "Multiple GPUs available, must run DDP tests"


# Tests use this fixture to parametrize over trainer names will now run with
# correct trainer, with unique ids for clarity in test reports.
@pytest.fixture(params=[trainer_name], ids=[trainer_name])
def trainer_name_param(request):
    return request.param
