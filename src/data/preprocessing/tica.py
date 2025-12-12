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

import logging

import deeptime as dt
import mdtraj as md
import numpy as np

SELECTION = "symbol == C or symbol == N or symbol == S"


def compute_distances(xyz):
    distance_matrix_ca = np.linalg.norm(xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1)
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def wrap(array):
    return (np.sin(array), np.cos(array))


def tica_features(trajectory, use_dihedrals=True, use_distances=True, selection=SELECTION):
    if trajectory.topology.n_residues == 8:
        logging.warning(
            "The 8AA TICA models no longer use the CA-only selection, aligning with the 2AA / 4AA models. "
        )
    trajectory = trajectory.atom_slice(trajectory.top.select(selection))
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_psi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate([*wrap(phi), *wrap(psi), *wrap(omega)], axis=-1)
    if use_distances:
        distances = compute_distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        return np.concatenate([distances, dihedrals], axis=-1)
    elif use_distances:
        return distances
    elif use_dihedrals:
        return dihedrals
    else:
        return []


def run_tica_heavy(trajectory, lagtime=100, dim=2):
    ca_features = tica_features(trajectory)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
    tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    return tica_model


def tica_features_ca(
    trajectory,
    ca_list,
):
    data = trajectory.xyz
    ca_pairs = [[ca_list[i], ca_list[j]] for i in range(len(ca_list)) for j in range(i + 1, len(ca_list))]
    coord_selected = data[:, ca_pairs, :]
    dist_vector_selected = coord_selected[:, :, 1, :] - coord_selected[:, :, 0, :]
    dist_selected = np.sqrt(np.square(dist_vector_selected).sum(axis=-1))  # C-alpha pairwise distance

    return dist_selected


def run_tica_ca(trajectory, topology, lagtime=500, dim=2):
    ca_list = [atom.index for atom in topology.atoms if atom.name == "CA"]
    ca_features = tica_features_ca(trajectory, ca_list=ca_list)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
    tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    return tica_model


def get_tica_model(data, topology):
    traj_samples = md.Trajectory(data, topology=topology)

    tica_model = run_tica_heavy(traj_samples, lagtime=100, dim=2)
    logging.info("Using all heavy atoms for TICA")

    return tica_model


class TicaModel:
    def __init__(
        self,
        projection,
        mean,
        dim=2,
    ):
        self.projection = projection
        self.mean = mean
        self.dim = dim

    def forward(self, x):
        X_centered = x - self.mean
        return X_centered @ self.projection[:, : self.dim]

    def transform(self, x):
        """Compatibility with original tica code from deeptime"""
        return self.forward(x)
