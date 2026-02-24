import hydra
import pytest
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

import logging

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import default_collate

from transferable_samplers.models.priors.normal_distribution import NormalDistribution
from transferable_samplers.data.transforms.padding import PaddingTransform

# Didn't see any slowdown for TarFlow with BF16 enabled, once had issues with invertibility
torch.set_float32_matmul_precision("highest")

# Constant parameters for all tests
NUM_SAMPLES = 32 # number of data samples to use per test
TEST_SEQUENCES = [
    "AA",
]
EXPERIMENT_OVERRIDE = [
    "experiment=training/transferable/prose_up_to_8aa",
    "data.num_aa_max=8",
    "++model.net.debug=True",
]

# Different network variants to test
NET_OVERRIDES = [
    ["model/net=prose"],
]

# Tolerances for the tests
INVERTIBILITY_ATOL = 5e-3  # any max pointwise error below this is acceptable
DLOGP_ATOL = 5e-3  # any samplewise dlogp error below this is acceptable


def move_to_device(data, device):
    """Recursively move nested dictionaries and tensors to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data


def get_test_data(datamodule, sequence: str):
    """Get test data using prepare_eval from the datamodule."""
    eval_ctx = datamodule.prepare_eval(sequence, stage="test")
    true_samples = eval_ctx.true_data.samples[:NUM_SAMPLES]
    system_cond = eval_ctx.system_cond

    # Build per-sample dicts for collation
    # system_cond has unbatched permutations/encodings (1D tensors)
    unpadded_data = []
    for i in range(len(true_samples)):
        sample = {
            "x": true_samples[i],
            "encodings": system_cond.encodings,
            "permutations": system_cond.permutations,
        }
        unpadded_data.append(sample)

    # Build padded versions
    padding_transform = PaddingTransform(datamodule.num_atoms)
    padded_data = []
    for sample in unpadded_data:
        padded_data.append(padding_transform(sample.copy()))

    return {"unpadded": unpadded_data, "padded": padded_data}


@pytest.fixture(scope="module")
def test_data():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train", overrides=EXPERIMENT_OVERRIDE)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    test_data_dict = {}
    for sequence in TEST_SEQUENCES:
        test_data_dict[sequence] = get_test_data(datamodule, sequence)

    yield test_data_dict

    GlobalHydra.instance().clear()


@pytest.fixture(scope="session", params=NET_OVERRIDES, ids=lambda p: ",".join(p))
def net(request):
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train", overrides=EXPERIMENT_OVERRIDE + request.param)

    assert cfg.model.net.debug, "Debug must be true if no checkpoint is provided"

    net = hydra.utils.instantiate(cfg.model.net)
    net.name = "no_ckpt"

    yield net

    GlobalHydra.instance().clear()


@torch.no_grad()
def test_invert_no_pad(net, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    for sequence in TEST_SEQUENCES:
        unpadded_data = test_data[sequence]["unpadded"]
        batch = default_collate(unpadded_data)

        x = batch["x"].to(device)
        permutations = move_to_device(batch["permutations"], device)
        encodings = move_to_device(batch["encodings"], device)

        z_pred, _ = net.forward(x, permutations=permutations, encodings=encodings)
        x_recon, _ = net.reverse(z_pred, permutations=permutations, encodings=encodings)

        max_abs_error = torch.max(torch.abs(x - x_recon)).item()
        assert max_abs_error < INVERTIBILITY_ATOL, (
            f"invertibility test (unpadded data) failed for sequence '{sequence}': "
            f"max_abs_error={max_abs_error:.6e} >= {INVERTIBILITY_ATOL}"
        )


@torch.no_grad()
def test_invert_from_prior_no_pad(net, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    for sequence in TEST_SEQUENCES:
        unpadded_data = test_data[sequence]["unpadded"]
        batch = default_collate(unpadded_data)

        x = batch["x"].to(device)
        permutations = move_to_device(batch["permutations"], device)
        encodings = move_to_device(batch["encodings"], device)

        prior = NormalDistribution(num_dimensions=x.shape[-1])

        z = prior.sample(num_samples=x.shape[0], num_atoms=x.shape[1]).to(device)

        x_pred, _ = net.reverse(z, permutations=permutations, encodings=encodings)
        batch_items_with_nan = torch.isnan(x_pred).any(dim=(1, 2))
        assert not batch_items_with_nan.sum().item()
        z_recon, _ = net.forward(x_pred, permutations=permutations, encodings=encodings)

        max_abs_error = torch.max(torch.abs(z - z_recon)).item()
        assert max_abs_error < INVERTIBILITY_ATOL, (
            f"invertibility test (unpadded data) failed for sequence '{sequence}': "
            f"max_abs_error={max_abs_error:.6e} >= {INVERTIBILITY_ATOL}"
        )


@torch.no_grad()
def test_fwd_pad_vs_no_pad(net, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    for sequence in TEST_SEQUENCES:
        unpadded_data = test_data[sequence]["unpadded"]
        padded_data = test_data[sequence]["padded"]
        batch_unpad = default_collate(unpadded_data)
        batch_pad = default_collate(padded_data)

        x_unpad = batch_unpad["x"].to(device)
        permutations_unpad = move_to_device(batch_unpad["permutations"], device)
        encodings_unpad = move_to_device(batch_unpad["encodings"], device)

        x_pad = batch_pad["x"].to(device)
        permutations_pad = move_to_device(batch_pad["permutations"], device)
        encodings_pad = move_to_device(batch_pad["encodings"], device)
        mask_pad = batch_pad["mask"].to(device)

        z_unpad, _ = net.forward(x_unpad, permutations=permutations_unpad, encodings=encodings_unpad)
        z_pad, _ = net.forward(x_pad, permutations=permutations_pad, encodings=encodings_pad, mask=mask_pad)

        z_sliced = z_pad[:, : x_unpad.shape[1]]

        max_abs_error = torch.max(torch.abs(z_unpad - z_sliced)).item()
        assert max_abs_error < INVERTIBILITY_ATOL, (
            f"fwd output consistency test between padded and unpadded data failed for sequence '{sequence}': "
            f"max_abs_error={max_abs_error:.6e} >= {INVERTIBILITY_ATOL}"
        )


@torch.no_grad()
def test_fwd_dlogp_no_pad(net, test_data):
    """Verify forward dlogp against true Jacobian dlogp, sample by sample."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    for sequence in TEST_SEQUENCES:
        unpadded_data = test_data[sequence]["unpadded"]
        for sample_idx, unpadded_sample in enumerate(unpadded_data):
            batch = default_collate([unpadded_sample])
            x = batch["x"].to(device)
            permutations = move_to_device(batch["permutations"], device)
            encodings = move_to_device(batch["encodings"], device)

            data_dim = x.shape[1] * x.shape[2]

            _, dlogp = net.forward(x, permutations=permutations, encodings=encodings)
            fwd_func = lambda x: net.forward(x, permutations=permutations, encodings=encodings)[0]
            fwd_jac = torch.autograd.functional.jacobian(fwd_func, x, vectorize=True)
            dlogp_true = torch.logdet(fwd_jac.view(data_dim, data_dim))
            dlogp_diff = abs(dlogp - dlogp_true).item()
            assert dlogp_diff < DLOGP_ATOL, (
                f"fwd dlogp test failed for sequence '{sequence}' "
                f"(sample {sample_idx}): dlogp_diff={dlogp_diff:.6e} >= {DLOGP_ATOL}"
            )


@torch.no_grad()
def test_fwd_dlogp_pad_vs_no_pad(net, test_data):
    """Verify forward dlogp is consistent between padded and unpadded inputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    for sequence in TEST_SEQUENCES:
        unpadded_data = test_data[sequence]["unpadded"]
        padded_data = test_data[sequence]["padded"]

        batch_unpad = default_collate(unpadded_data)
        batch_pad = default_collate(padded_data)
        x_unpad = batch_unpad["x"].to(device)
        permutations_unpad = move_to_device(batch_unpad["permutations"], device)
        encodings_unpad = move_to_device(batch_unpad["encodings"], device)

        x_pad = batch_pad["x"].to(device)
        permutations_pad = move_to_device(batch_pad["permutations"], device)
        encodings_pad = move_to_device(batch_pad["encodings"], device)
        mask_pad = batch_pad["mask"].to(device)

        _, dlogp_unpad = net.forward(x_unpad, permutations=permutations_unpad, encodings=encodings_unpad)
        _, dlogp_pad = net.forward(x_pad, permutations=permutations_pad, encodings=encodings_pad, mask=mask_pad)

        max_abs_error = torch.max(torch.abs(dlogp_unpad - dlogp_pad)).item()
        assert max_abs_error < DLOGP_ATOL, (
            f"fwd dlogp consistency test between padded and unpadded data failed for sequence '{sequence}' "
            f"max_abs_error={max_abs_error:.6e} >= {DLOGP_ATOL}"
        )


@torch.no_grad()
def test_fwd_vs_rev_dlogp_no_pad(net, test_data):
    """Verify that fwd_dlogp + rev_dlogp = 0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    for sequence in TEST_SEQUENCES:
        unpadded_data = test_data[sequence]["unpadded"]
        batch = default_collate(unpadded_data)

        x = batch["x"].to(device)
        permutations = move_to_device(batch["permutations"], device)
        encodings = move_to_device(batch["encodings"], device)

        z_pred, fwd_dlogp = net.forward(x, permutations=permutations, encodings=encodings)
        _, rev_dlogp = net.reverse(z_pred, permutations=permutations, encodings=encodings)

        max_abs_error = torch.max(torch.abs(fwd_dlogp + rev_dlogp)).item()
        assert max_abs_error < DLOGP_ATOL, (
            f"fwd vs rev dlogp test (unpadded data) failed for sequence '{sequence}' "
        )
