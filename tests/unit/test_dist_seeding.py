"""Tests for the DDP rank-local RNG seeding scheme.

Verifies the ``seed_rank_local()`` mechanism and its integration into
``BaseLightningModule.setup()``.

Three groups:
  A. Unit tests for ``seed_rank_local()`` mechanism (non-distributed, mocked rank).
  B. DDP integration tests (real gloo, 2 ranks) — bug, fix, reproducibility.
  C. ``BaseLightningModule.setup()`` integration — call verification, idempotency.
"""

import json
import os
import random
import shutil
import socket
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lightning import seed_everything

from transferable_samplers.models.base_lightning_module import BaseLightningModule
from transferable_samplers.utils import dist_utils

# ======================================================================
# Helpers
# ======================================================================


def _find_free_port() -> str:
    """Return a free TCP port as a string (for gloo rendezvous).

    ``SO_REUSEADDR`` avoids lingering ``TIME_WAIT`` sockets binding the port
    on rapid re-runs (e.g. when the gloo store is retried after a flaky spawn).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
    elif strategy == "seed_rank_local":
        seed_everything(base_seed, workers=True)
        dist_utils.seed_rank_local()
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
    """Spawn DDP workers with a given strategy and return per-rank RNG draws.

    Retries once on a fresh port if any worker exits non-zero — gloo rendezvous
    is occasionally flaky under fork+``mp.spawn`` in CI/concurrent test runs.
    """
    ctx = mp.get_context("fork")
    last_err: Exception | None = None
    for _attempt in range(2):
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
        failed = False
        for p in procs:
            p.join(timeout=60)
            if p.exitcode != 0:
                failed = True
                last_err = RuntimeError(f"DDP worker exited with code {p.exitcode}")
                # Terminate any stragglers so the next attempt starts clean.
                for q in procs:
                    if q.is_alive():
                        q.terminate()
                break
        if not failed:
            results = []
            for rank in range(world_size):
                rank_path = Path(tmpdir) / f"rank_{rank}.json"
                with rank_path.open() as f:
                    results.append(json.load(f))
            shutil.rmtree(tmpdir)
            return results
        shutil.rmtree(tmpdir, ignore_errors=True)
    # Exhausted retries — surface the last failure.
    assert last_err is not None
    raise last_err


# ======================================================================
# Part A: Unit tests for seed_rank_local() mechanism
# ======================================================================


@pytest.mark.essential
class TestSeedRankLocalMechanism:
    """Unit tests for seed_rank_local() (non-distributed, mocked rank)."""

    def test_noop_when_not_distributed(self):
        """seed_rank_local is a no-op when dist is not initialized (rank defaults to 0)."""
        torch.manual_seed(42)
        before = torch.get_rng_state()
        dist_utils.seed_rank_local()
        after = torch.get_rng_state()
        assert before.equal(after)

    def test_noop_on_rank_zero(self):
        """Explicit rank 0 is a no-op (already correctly seeded by seed_everything)."""
        torch.manual_seed(42)
        before = torch.get_rng_state()
        with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=0):
            dist_utils.seed_rank_local()
        after = torch.get_rng_state()
        assert before.equal(after)

    def test_reseeds_torch_with_base_seed_plus_rank(self):
        """Rank N gets re-seeded with base_seed + N on torch."""
        with (
            patch("transferable_samplers.utils.dist_utils.get_rank", return_value=2),
            patch.dict(os.environ, {"PL_GLOBAL_SEED": "100"}),
        ):
            dist_utils.seed_rank_local()
            post_seed = torch.randn(4)

        torch.manual_seed(102)  # 100 + 2
        expected = torch.randn(4)

        assert torch.equal(post_seed, expected)

    def test_reseeds_numpy_with_base_seed_plus_rank(self):
        """Rank N gets re-seeded with base_seed + N on numpy."""
        with (
            patch("transferable_samplers.utils.dist_utils.get_rank", return_value=3),
            patch.dict(os.environ, {"PL_GLOBAL_SEED": "50"}),
        ):
            dist_utils.seed_rank_local()
            post_seed = np.random.rand(4).tolist()

        np.random.seed(53)  # 50 + 3
        expected = np.random.rand(4).tolist()

        assert post_seed == expected

    def test_reseeds_python_random_with_base_seed_plus_rank(self):
        """Rank N gets re-seeded with base_seed + N on python.random."""
        with (
            patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1),
            patch.dict(os.environ, {"PL_GLOBAL_SEED": "77"}),
        ):
            dist_utils.seed_rank_local()
            post_seed = [random.random() for _ in range(4)]

        random.seed(78)  # 77 + 1
        expected = [random.random() for _ in range(4)]

        assert post_seed == expected

    def test_torch_draws_differ_across_ranks(self):
        """Rank 0 and rank 1 produce different torch draws after seeding."""
        with patch.dict(os.environ, {"PL_GLOBAL_SEED": "42"}):
            with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1):
                dist_utils.seed_rank_local()
                rank1_draw = torch.randn(8)

            torch.manual_seed(42)  # rank 0
            rank0_draw = torch.randn(8)

        assert not torch.equal(rank0_draw, rank1_draw)

    def test_no_substream_collision(self):
        """Per-rank seeds produce independent streams with no collision risk.

        Unlike offset-based drift, rank 0 can draw indefinitely without ever
        colliding with rank 1's stream — they use independent seeds.
        """
        with patch.dict(os.environ, {"PL_GLOBAL_SEED": "42"}):
            torch.manual_seed(42)  # rank 0
            _ = torch.randn(10_000_000)  # way past any drift STRIDE budget
            rank0_after = torch.randn(1)

            with patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1):
                dist_utils.seed_rank_local()
                rank1_first = torch.randn(1)

        assert not torch.equal(rank0_after, rank1_first), (
            "Per-rank seeds produce independent streams — no sub-stream collision."
        )

    def test_reproducible_same_seed_same_rank(self):
        """Same base seed + same rank produces identical draws across runs."""

        def _run():
            torch.manual_seed(999)  # arbitrary pre-state
            with (
                patch("transferable_samplers.utils.dist_utils.get_rank", return_value=3),
                patch.dict(os.environ, {"PL_GLOBAL_SEED": "42"}),
            ):
                dist_utils.seed_rank_local()
            return torch.randn(8)

        assert torch.equal(_run(), _run())

    def test_idempotent_multiple_calls(self):
        """Calling seed_rank_local twice produces the same state (no compounding).

        This is the key advantage over offset-based drift: setup() may be called
        multiple times (fit + validate + test) and the RNG state is deterministic
        regardless of how many stages ran.
        """

        def _run_two_calls():
            torch.manual_seed(999)  # arbitrary pre-state
            with (
                patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1),
                patch.dict(os.environ, {"PL_GLOBAL_SEED": "42"}),
            ):
                dist_utils.seed_rank_local()
                dist_utils.seed_rank_local()  # second call
            return torch.randn(4)

        def _run_one_call():
            torch.manual_seed(999)  # same pre-state
            with (
                patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1),
                patch.dict(os.environ, {"PL_GLOBAL_SEED": "42"}),
            ):
                dist_utils.seed_rank_local()
            return torch.randn(4)

        assert torch.equal(_run_two_calls(), _run_one_call()), (
            "seed_rank_local should be idempotent: two calls produce the same state as one."
        )

    def test_cuda_manual_seed_all_when_available(self):
        """When CUDA is available, seed_rank_local calls torch.cuda.manual_seed_all.

        torch.manual_seed is mocked to isolate our code's CUDA branch (it
        internally calls manual_seed_all regardless of is_available).
        """
        with (
            patch("transferable_samplers.utils.dist_utils.get_rank", return_value=2),
            patch.dict(os.environ, {"PL_GLOBAL_SEED": "100"}),
            patch("torch.manual_seed"),  # no-op to prevent internal cuda call
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed_all") as mock_cuda_seed,
        ):
            dist_utils.seed_rank_local()

        mock_cuda_seed.assert_called_once_with(102)  # 100 + 2

    def test_cuda_skipped_when_unavailable(self):
        """When CUDA is unavailable, manual_seed_all is not called by our code."""
        with (
            patch("transferable_samplers.utils.dist_utils.get_rank", return_value=2),
            patch.dict(os.environ, {"PL_GLOBAL_SEED": "100"}),
            patch("torch.manual_seed"),  # no-op to prevent internal cuda call
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.manual_seed_all") as mock_cuda_seed,
        ):
            dist_utils.seed_rank_local()

        mock_cuda_seed.assert_not_called()


# ======================================================================
# Part B: DDP integration tests (real gloo, 2 ranks)
# ======================================================================


@pytest.mark.essential
class TestDDPSeeding:
    """Integration tests with real gloo DDP (2 ranks).

    Tests the bug (identical draws across ranks) and the seed_rank_local fix
    (all three RNG sources: torch, numpy, python.random).
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

    def test_fix_torch(self):
        """Fix: seed_rank_local decorrelates torch draws across ranks."""
        r = _run_ddp("seed_rank_local")
        assert r[0]["torch"] != r[1]["torch"]

    def test_fix_numpy(self):
        """Fix: seed_rank_local decorrelates numpy draws across ranks."""
        r = _run_ddp("seed_rank_local")
        assert r[0]["numpy"] != r[1]["numpy"]

    def test_fix_python_random(self):
        """Fix: seed_rank_local decorrelates python.random draws across ranks."""
        r = _run_ddp("seed_rank_local")
        assert r[0]["python"] != r[1]["python"]

    def test_reproducible_across_runs(self):
        """Same seed → same per-rank draws across independent runs."""
        r1 = _run_ddp("seed_rank_local")
        r2 = _run_ddp("seed_rank_local")
        assert r1[0]["torch"] == r2[0]["torch"]
        assert r1[1]["torch"] == r2[1]["torch"]

    def test_no_substream_collision(self):
        """Per-rank seeds: independent streams, no sub-stream collision risk.

        Unlike offset-based drift (where rank 0's (STRIDE+1)-th draw == rank 1's
        first draw), per-rank seeds produce fully independent streams with no
        boundary.
        """
        seed_everything(42)  # rank 0
        _ = torch.randn(10_000_000)  # way past any drift budget
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
    """Tests for BaseLightningModule.setup() calling seed_rank_local."""

    def test_setup_calls_seed_rank_local_for_every_stage(self):
        """setup() calls seed_rank_local once per stage invocation."""
        mod = _make_module()
        with (
            patch.object(BaseLightningModule, "trainer", new_callable=PropertyMock, return_value=None),
            patch("transferable_samplers.models.base_lightning_module.seed_rank_local") as mock_seed,
        ):
            for stage in ("fit", "validate", "test", "predict"):
                mod.setup(stage)
        assert mock_seed.call_count == 4

    def test_seed_rank_local_idempotent_across_setup_calls(self):
        """Idempotency: calling setup() multiple times does NOT compound.

        seed_rank_local re-seeds from a fixed (base_seed + rank) each time, so
        two setup() calls produce the same RNG state as one call. This avoids
        the cross-stage reproducibility footgun of offset-based drift.
        """

        def _run_two_calls():
            torch.manual_seed(999)  # arbitrary pre-state
            with (
                patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1),
                patch.dict(os.environ, {"PL_GLOBAL_SEED": "42"}),
            ):
                dist_utils.seed_rank_local()  # setup("fit")
                dist_utils.seed_rank_local()  # setup("test")
            return torch.randn(4)

        def _run_one_call():
            torch.manual_seed(999)  # same pre-state
            with (
                patch("transferable_samplers.utils.dist_utils.get_rank", return_value=1),
                patch.dict(os.environ, {"PL_GLOBAL_SEED": "42"}),
            ):
                dist_utils.seed_rank_local()  # setup("test") only
            return torch.randn(4)

        assert torch.equal(_run_two_calls(), _run_one_call()), (
            "seed_rank_local should NOT compound: two setup() calls produce the same RNG state as one call."
        )
