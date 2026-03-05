# -------------------------------------------------------------------------
# Adapted from
# https://osf.io/n8vz3/
# Licensed under Creative Commons Attribution 4.0 International
# -------------------------------------------------------------------------
# Copyright (c) 2024 Leon Klein, Frank Noé.
# https://creativecommons.org/licenses/by/4.0/
# -------------------------------------------------------------------------
# Modifications Copyright (c) 2025 transferable-samplers contributors
# Licensed under the MIT License (see LICENSE in the repository root).
# -------------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import Any

import deeptime as dt
import mdtraj as md
import numpy as np
import torch

SELECTION = "symbol == C or symbol == N or symbol == S"


def _compute_distances(xyz: np.ndarray) -> np.ndarray:
    distance_matrix_ca = np.linalg.norm(xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1)
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def wrap(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrap angles into (sin, cos) pairs."""
    return (np.sin(array), np.cos(array))


def tica_features(
    trajectory: md.Trajectory, use_dihedrals: bool = True, use_distances: bool = True, selection: str = SELECTION
) -> np.ndarray | list[Any]:
    """Compute TICA input features (distances and/or wrapped dihedrals) from a trajectory."""
    if trajectory.topology.n_residues == 8:
        logging.warning("The 8AA TICA models no longer use the CA-only selection, aligning with the 2AA / 4AA models.")
    trajectory = trajectory.atom_slice(trajectory.top.select(selection))
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_psi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate([*wrap(phi), *wrap(psi), *wrap(omega)], axis=-1)
    if use_distances:
        distances = _compute_distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        # pyrefly: ignore [unbound-name]
        return np.concatenate([distances, dihedrals], axis=-1)
    elif use_distances:
        # pyrefly: ignore [unbound-name]
        return distances
    elif use_dihedrals:
        # pyrefly: ignore [unbound-name]
        return dihedrals
    else:
        return []


class TicaModel:
    """Lightweight TICA model that projects features using a precomputed projection matrix.

    Args:
        projection: TICA projection matrix (n_features, max_dim).
        mean: Feature mean for centering.
        dim: Number of TICA dimensions to retain.
    """

    def __init__(
        self,
        projection: np.ndarray,
        mean: np.ndarray,
        dim: int | np.ndarray = 2,
    ) -> None:
        self.projection = projection
        self.mean = mean
        self.dim = dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project data onto the TICA components."""
        x_centered = x - self.mean
        return x_centered @ self.projection[:, : self.dim]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Compatibility with original tica code from deeptime."""
        return self.forward(x)


def _run_tica_cns(trajectory: md.Trajectory, lagtime: int = 100, dim: int = 2) -> Any:
    """Fit a Koopman-reweighted TICA model on C/N/S atom features."""
    ca_features = tica_features(trajectory)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
    tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    return tica_model


def get_tica_model(data: torch.Tensor, topology: md.Topology) -> Any:
    """Build a TICA model from trajectory data and topology."""
    traj_samples = md.Trajectory(data, topology=topology)

    tica_model = _run_tica_cns(traj_samples, lagtime=100, dim=2)
    logging.info("Using all C,N,S atoms for TICA")

    return tica_model
