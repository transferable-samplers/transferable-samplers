from __future__ import annotations

import math
from typing import Any, Literal

import mdtraj as md
import numpy as np
import ot as pot
import torch

from transferable_samplers.data.preprocessing.tica import tica_features


def _wasserstein_distance(x0: torch.Tensor, x1: torch.Tensor, power: Literal[1, 2] = 2) -> float:
    """Wasserstein distance between two point clouds.

    Args:
        x0: First point cloud [n, d] or [n, ...].
        x1: Second point cloud [m, d] or [m, ...].
        power: 1 for W1, 2 for W2.
    """
    assert power == 1 or power == 2
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    cost = torch.cdist(x0, x1)
    if power == 2:
        cost = cost**2
    ret = pot.emd2(a, b, cost.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        # pyrefly: ignore [bad-argument-type]
        ret = math.sqrt(ret)
    return ret


def _torus_wasserstein_distance(
    x0: np.ndarray | torch.Tensor, x1: np.ndarray | torch.Tensor, power: Literal[1, 2] = 2
) -> float:
    """Wasserstein distance on the torus (circular/wrapped distances).

    Args:
        x0: First angle array.
        x1: Second angle array.
        power: 1 for W1, 2 for W2.
    """
    assert power == 1 or power == 2
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    x0 = x0[:, None]
    x1 = x1[None, :]
    dists = np.minimum(np.abs(x0 - x1), 2 * np.pi - np.abs(x0 - x1))
    if power == 2:
        dists = dists**2
    ret = pot.emd2(a, b, dists.sum(-1), numItermax=int(1e9))
    if power == 2:
        # pyrefly: ignore [no-matching-overload]
        ret = np.sqrt(ret).item()
    else:
        # pyrefly: ignore [bad-argument-type]
        ret = float(ret)
    return ret


def _get_phi_psi_vectors(samples: torch.Tensor, topology: md.Topology) -> tuple[np.ndarray, np.ndarray]:
    """Extract phi/psi dihedral angles from samples."""
    samples = samples.cpu()
    traj = md.Trajectory(samples, topology=topology)
    phis = md.compute_phi(traj)[1]
    psis = md.compute_psi(traj)[1]
    return phis, psis


def energy_wasserstein(pred_energy: torch.Tensor, true_energy: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """Compute energy W1 and W2 distances between predicted and true energy distributions."""
    pred = pred_energy.cpu().numpy()
    true = true_energy.cpu().numpy()
    return {
        # pyrefly: ignore [bad-argument-type]
        f"{prefix}/energy-w2": math.sqrt(pot.emd2_1d(true, pred)),
        f"{prefix}/energy-w1": pot.emd2_1d(true, pred, metric="euclidean"),
    }


def torus_wasserstein(
    true_samples: torch.Tensor, pred_samples: torch.Tensor, topology: md.Topology, prefix: str = ""
) -> dict[str, float]:
    """Compute torus Wasserstein W2 on phi/psi dihedral angles."""
    phis_true, psis_true = _get_phi_psi_vectors(true_samples, topology)
    x_true = torch.cat([torch.from_numpy(phis_true), torch.from_numpy(psis_true)], dim=1)

    phis_pred, psis_pred = _get_phi_psi_vectors(pred_samples, topology)
    x_pred = torch.cat([torch.from_numpy(phis_pred), torch.from_numpy(psis_pred)], dim=1)

    return {
        f"{prefix}/torus-w2": _torus_wasserstein_distance(x_true, x_pred, power=2),
    }


def tica_wasserstein(
    true_samples: torch.Tensor, pred_samples: torch.Tensor, topology: md.Topology, tica_model: Any, prefix: str = ""
) -> dict[str, float]:
    """Compute Wasserstein W2 on TICA coordinates."""
    true_traj = md.Trajectory(true_samples.cpu().numpy(), topology=topology)
    pred_traj = md.Trajectory(pred_samples.cpu().numpy(), topology=topology)

    features_true = tica_features(true_traj)
    features_pred = tica_features(pred_traj)

    n = min(len(features_true), len(features_pred))
    tics_true = torch.Tensor(tica_model.transform(features_true))[:n, 0:2]
    tics_pred = torch.Tensor(tica_model.transform(features_pred))[:n, 0:2]

    return {
        f"{prefix}/tica-w2": _wasserstein_distance(tics_true, tics_pred, power=2),
    }
