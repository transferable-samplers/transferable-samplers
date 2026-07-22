"""Tests for the DDP rank-local RNG seeding scheme (PR #38).

Verifies the seeding mechanism introduced in ``drift_rng_state()`` and called
from ``BaseLightningModule.setup()``.

Three groups:
  A. Unit tests for ``drift_rng_state()`` mechanism (non-distributed, mocked rank).
  B. DDP integration tests (real gloo, 2 ranks) — bug, fix, gaps, alternative.
  C. ``BaseLightningModule.setup()`` integration — call verification, compounding.
"""

import json
import os
import random
import shutil
import socket
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lightning import seed_everything

from transferable_samplers.models.base_lightning_module import BaseLightningModule
from transferable_samplers.utils import dist_utils

STRIDE = 1 << 20  # must match drift_rng_state's internal STRIDE


# ======================================================================
# Helpers
# ======================================================================


def _find_free_port() -> str:
    """Return a free TCP port as a string (for gloo rendezvous)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return str(s.getsockname()[1])


def _ddp_worker(
    rank: int,
    world_size: int,
    port: str,
    strategy: str,
    base_seed: int,
    tmpdir: str,
    n_draws: int,
) -> None:
    """Module-level DDP worker: apply a seeding strategy, draw from all RNGs, write to file."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if strategy == "seed_only":
        seed_everything(base_seed, workers=False)
    elif strategy == "seed_workers":
        seed_everything(base_seed, workers=True)
    elif strategy == "drift":
        seed_everything(base_seed, workers=False)
        dist_utils.drift_rng_state()
    elif strategy == "per_rank_seed":
        seed_everything(base_seed + rank, workers=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    torch_draw = torch.randn(n_draws).tolist()
    np_draw = np.random.rand(n_draws).tolist()
    py_draw = [random.random() for _ in range(n_draws)]

    result = {"torch": torch_draw, "numpy": np_draw, "python": py_draw, "rank": rank}
    out_path = Path(tmpdir) / f"rank_{rank}.json"
    with out_path.open("w") as f:
        json.dump(result, f)

    dist.barrier()
    dist.destroy_process_group()


def _run_ddp(strategy: str, world_size: int = 2, base_seed: int = 42, n_draws: int = 8) -> list[dict]:
    """Spawn DDP workers with a given strategy and return per-rank RNG draws."""
    ctx = mp.get_context("fork")
    port = _find_free_port()
    tmpdir = tempfile.mkdtemp()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_ddp_worker,
            args=(rank, world_size, port, strategy, base_seed, tmpdir, n_draws),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=60)
        if p.exitcode != 0:
            raise RuntimeError(f"DDP worker exited with code {p.exitcode}")
    results = []
    for rank in range(world_size):
        rank_path = Path(tmpdir) / f"rank_{rank}.json"
        with rank_path.open() as f:
            results.append(json.load(f))
    shutil.rmtree(tmpdir)
    return results


def _per_rank_seed(base_seed: int) -> None:
    """Alternative strategy: per-rank derived seed (fixes torch + numpy + python)."""
    seed_everything(base_seed + dist_utils.get_rank(), workers=True)


# ======================================================================
# Part A: Unit tests for drift_rng_state() mechanism
# ======================================================================


@pytest.mark.essential
class TestDriftMechanism:
    """Unit tests for drift_rng_state() CPU/CUDA advancement (non-distributed, mocked rank)."""

    def test_noop_when_not_distributed(self):
        """drift_rng_state is a no-op when dist is not initialized (rank defaults to 0)."""
        torch.manual_seed(42)
        before = torch.get_rng_state()
        dist_utils.drift_rng_state()
        after = torch.get_rng_state()
        assert before.equal(after)

    def test_noop_on_rank_zero(self):
        """Explicit rank 0 is a no-op."""
        torch.manual_seed(42)
        before = torch.get_rng_state()
        with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=0):
            dist_utils.drift_rng_state()
        after = torch.get_rng_state()
        assert before.equal(after)

    def test_cpu_rng_advances_by_rank_times_stride(self):
        """Rank N drift advances CPU RNG by exactly N*STRIDE elements."""
        with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=2):
            torch.manual_seed(42)
            dist_utils.drift_rng_state()
            post_drift = torch.randn(4)

        torch.manual_seed(42)
        _ = torch.randn(2 * STRIDE)
        expected = torch.randn(4)

        assert torch.equal(post_drift, expected)

    def test_rank1_stream_equals_manual_burn(self):
        """Rank 1 post-drift draw == manual burn of STRIDE elements from same seed.

        Verifies sub-stream continuity: rank 1 picks up right after rank 0's STRIDE budget.
        """
        with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1):
            torch.manual_seed(42)
            dist_utils.drift_rng_state()
            drifted_draw = torch.randn(4)

        torch.manual_seed(42)
        _ = torch.randn(STRIDE)
        manual_draw = torch.randn(4)

        assert torch.equal(drifted_draw, manual_draw)

    def test_cpu_draws_differ_across_ranks(self):
        """Rank 0 and rank 1 produce different CPU draws after drift."""
        with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1):
            torch.manual_seed(42)
            dist_utils.drift_rng_state()
            rank1_draw = torch.randn(8)

        torch.manual_seed(42)
        rank0_draw = torch.randn(8)  # rank 0 / not distributed → no drift

        assert not torch.equal(rank0_draw, rank1_draw)

    def test_reproducible_same_seed_same_rank(self):
        """Same seed + same rank produces identical draws across runs."""

        def _run():
            torch.manual_seed(42)
            with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=3):
                dist_utils.drift_rng_state()
            return torch.randn(8)

        assert torch.equal(_run(), _run())

    def test_stride_budget_limitation(self):
        """Document: rank 0's (STRIDE+1)-th draw collides with rank 1's first draw.

        If any rank consumes more than STRIDE RNG draws, it enters the next rank's
        sub-stream. This is the fundamental limitation of offset-based drift vs.
        independent per-rank seeds.
        """
        with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1):
            torch.manual_seed(42)
            dist_utils.drift_rng_state()
            rank1_first = torch.randn(1)

        # From the same seed, burn STRIDE then draw 1 — this is rank 1's first draw
        torch.manual_seed(42)
        _ = torch.randn(STRIDE)
        rank0_after_stride = torch.randn(1)

        assert torch.equal(rank1_first, rank0_after_stride), (
            "Rank 0's (STRIDE+1)-th draw == rank 1's first draw. "
            "If rank 0 exceeds its STRIDE budget, sub-streams collide."
        )

    def test_cuda_offset_set_when_available(self):
        """When CUDA is available, drift advances the generator offset by rank*STRIDE."""
        mock_gen = MagicMock()
        mock_gen.get_offset.return_value = 1000

        with (
            patch("transferable_samplers.utils.dist_utils.get_rank", return_value=2),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.default_generators", new=[mock_gen]),
        ):
            dist_utils.drift_rng_state()

        mock_gen.get_offset.assert_called_once()
        mock_gen.set_offset.assert_called_once_with(1000 + 2 * STRIDE)

    def test_cuda_skipped_when_unavailable(self):
        """When CUDA is unavailable, no generator offset is set."""
        mock_gen = MagicMock()

        with (
            patch("transferable_samplers.utils.dist_utils.get_rank", return_value=2),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.default_generators", new=[mock_gen]),
        ):
            dist_utils.drift_rng_state()

        mock_gen.set_offset.assert_not_called()


# ======================================================================
# Part B: DDP integration tests (real gloo, 2 ranks)
# ======================================================================


@pytest.mark.essential
class TestDDPSeeding:
    """Integration tests with real gloo DDP (2 ranks).

    Tests the bug (identical draws across ranks), the drift fix (torch-only),
    its gaps (numpy/python untouched), and the per-rank-seed alternative.
    """

    def test_bug_seed_only_ranks_identical(self):
        """Bug: seed_everything without workers → all ranks produce identical draws."""
        r = _run_ddp("seed_only")
        assert r[0]["torch"] == r[1]["torch"]
        assert r[0]["numpy"] == r[1]["numpy"]
        assert r[0]["python"] == r[1]["python"]

    def test_bug_seed_workers_ranks_identical(self):
        """Bug: workers=True does NOT re-seed the main-process RNG per rank."""
        r = _run_ddp("seed_workers")
        assert r[0]["torch"] == r[1]["torch"]
        assert r[0]["numpy"] == r[1]["numpy"]
        assert r[0]["python"] == r[1]["python"]

    def test_drift_fixes_torch(self):
        """Fix: drift_rng_state decorrelates torch CPU draws across ranks."""
        r = _run_ddp("drift")
        assert r[0]["torch"] != r[1]["torch"]

    def test_drift_does_not_fix_numpy(self):
        """Gap: drift does NOT decorrelate numpy RNG across ranks.

        drift_rng_state only advances torch's CPU/CUDA generators. numpy's
        independent RNG remains seeded identically across ranks.
        """
        r = _run_ddp("drift")
        assert r[0]["numpy"] == r[1]["numpy"]

    def test_drift_does_not_fix_python_random(self):
        """Gap: drift does NOT decorrelate python.random across ranks."""
        r = _run_ddp("drift")
        assert r[0]["python"] == r[1]["python"]

    def test_per_rank_seed_fixes_torch(self):
        """Alternative: per-rank seed decorrelates torch draws."""
        r = _run_ddp("per_rank_seed")
        assert r[0]["torch"] != r[1]["torch"]

    def test_per_rank_seed_fixes_numpy(self):
        """Alternative: per-rank seed decorrelates numpy draws."""
        r = _run_ddp("per_rank_seed")
        assert r[0]["numpy"] != r[1]["numpy"]

    def test_per_rank_seed_fixes_python(self):
        """Alternative: per-rank seed decorrelates python.random draws."""
        r = _run_ddp("per_rank_seed")
        assert r[0]["python"] != r[1]["python"]

    def test_drift_reproducible_across_runs(self):
        """Drift: same seed → same per-rank draws across independent runs."""
        r1 = _run_ddp("drift")
        r2 = _run_ddp("drift")
        assert r1[0]["torch"] == r2[0]["torch"]
        assert r1[1]["torch"] == r2[1]["torch"]

    def test_per_rank_seed_reproducible_across_runs(self):
        """Per-rank seed: same seed → same per-rank draws across independent runs."""
        r1 = _run_ddp("per_rank_seed")
        r2 = _run_ddp("per_rank_seed")
        assert r1[0]["torch"] == r2[0]["torch"]
        assert r1[1]["torch"] == r2[1]["torch"]

    def test_per_rank_seed_no_substream_collision(self):
        """Per-rank seed: independent seeds, no STRIDE sub-stream collision risk.

        Unlike drift (where rank 0's (STRIDE+1)-th draw == rank 1's first draw),
        per-rank seeds produce fully independent streams with no boundary.
        """
        seed_everything(42)  # rank 0
        _ = torch.randn(STRIDE + 100)  # way past the drift budget
        rank0_after = torch.randn(1)

        seed_everything(43)  # rank 1
        rank1_first = torch.randn(1)

        assert not torch.equal(rank0_after, rank1_first), (
            "Per-rank seeds produce independent streams — no sub-stream collision."
        )


# ======================================================================
# Part C: BaseLightningModule.setup() integration
# ======================================================================


class _MinimalLightningModule(BaseLightningModule):
    """Minimal concrete subclass for testing setup()."""

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def compute_primary_loss(self, batch):
        return torch.tensor(0.0)

    def generate_proposal(self, net, num_samples, system_cond):
        return torch.zeros(num_samples, 1, 3), torch.zeros(num_samples)

    def proposal_energy(self, net, x, system_cond):
        return torch.zeros(x.shape[0])


def _make_module() -> _MinimalLightningModule:
    return _MinimalLightningModule(
        net=torch.nn.Identity(),
        optimizer=partial(torch.optim.SGD, lr=0.0),
        prior=None,
    )


@pytest.mark.essential
class TestSetupIntegration:
    """Tests for BaseLightningModule.setup() calling drift_rng_state."""

    def test_setup_calls_drift_for_every_stage(self):
        """setup() calls drift_rng_state once per stage invocation."""
        mod = _make_module()
        with (
            patch.object(BaseLightningModule, "trainer", new_callable=PropertyMock, return_value=None),
            patch("transferable_samplers.models.base_lightning_module.drift_rng_state") as mock_drift,
        ):
            for stage in ("fit", "validate", "test", "predict"):
                mod.setup(stage)
        assert mock_drift.call_count == 4

    def test_drift_compounds_across_setup_calls(self):
        """Footgun: calling drift twice (setup fit + setup test) advances RNG twice.

        Since drift advances from the *current* state (not from a fixed seed),
        calling setup() per stage means the RNG offset depends on how many stages
        ran before. Two setup() calls produce different draws than one.
        """
        with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1):
            # Scenario A: setup("fit") then setup("test") — drift called twice
            torch.manual_seed(42)
            dist_utils.drift_rng_state()  # setup("fit")
            dist_utils.drift_rng_state()  # setup("test")
            draw_two_calls = torch.randn(4)

            # Scenario B: setup("test") only — drift called once
            torch.manual_seed(42)
            dist_utils.drift_rng_state()  # setup("test")
            draw_one_call = torch.randn(4)

        assert not torch.equal(draw_two_calls, draw_one_call), (
            "Drift compounds: two setup() calls produce different RNG state than one. "
            "This is the cross-stage reproducibility footgun."
        )

    def test_per_rank_seed_once_no_compounding(self):
        """Alternative: per-rank seed with one-time guard does NOT compound.

        Calling setup() multiple times is safe because the second call is a no-op
        — the seed is only applied once, producing a deterministic state regardless
        of how many stages ran.
        """
        seeded = [False]

        def seed_once(base_seed):
            if not seeded[0]:
                _per_rank_seed(base_seed)
                seeded[0] = True

        with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1):
            # Scenario A: setup("fit") then setup("test") — seed fires once
            seeded[0] = False
            torch.manual_seed(999)  # arbitrary pre-state
            seed_once(42)  # setup("fit")
            seed_once(42)  # setup("test") — no-op
            draw_two_calls = torch.randn(4)

            # Scenario B: setup("test") only — seed fires once
            seeded[0] = False
            torch.manual_seed(999)  # same pre-state
            seed_once(42)  # setup("test")
            draw_one_call = torch.randn(4)

        assert torch.equal(draw_two_calls, draw_one_call), (
            "Per-rank seed-once should NOT compound: two setup() calls produce the same RNG state as one call."
        )
