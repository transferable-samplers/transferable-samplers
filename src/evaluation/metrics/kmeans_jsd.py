import mdtraj as md
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans

from src.data.preprocessing.tica import tica_features, wrap


def _kmeans_jsd(true_features, pred_features, n_clusters=20):
    """Compute JSD between two feature sets via KMeans discretization."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(true_features)
    true_states = kmeans.labels_
    pred_states = kmeans.predict(pred_features)

    epsilon = 1e-10
    true_counts = np.bincount(true_states, minlength=n_clusters)
    pred_counts = np.bincount(pred_states, minlength=n_clusters)
    true_dist = (true_counts + epsilon) / (true_counts.sum() + n_clusters * epsilon)
    pred_dist = (pred_counts + epsilon) / (pred_counts.sum() + n_clusters * epsilon)

    return jensenshannon(true_dist, pred_dist, base=2) ** 2


def _compute_dihedrals(trajectory):
    """Compute sin/cos-wrapped phi, psi, omega dihedrals."""
    _, phi = md.compute_phi(trajectory)
    _, psi = md.compute_psi(trajectory)
    _, omega = md.compute_omega(trajectory)
    return np.concatenate([*wrap(phi), *wrap(psi), *wrap(omega)], axis=-1)


def tica_kmeans_jsd(true_samples, pred_samples, topology, tica_model, n_clusters=20, prefix=""):
    """JSD on KMeans-discretized TICA coordinates."""
    true_traj = md.Trajectory(true_samples.cpu().numpy(), topology=topology)
    pred_traj = md.Trajectory(pred_samples.cpu().numpy(), topology=topology)

    features_true = tica_features(true_traj)
    features_pred = tica_features(pred_traj)

    n = min(len(features_true), len(features_pred))
    tics_true = torch.Tensor(tica_model.transform(features_true))[:n, 0:2]
    tics_pred = torch.Tensor(tica_model.transform(features_pred))[:n, 0:2]

    jsd = _kmeans_jsd(tics_true, tics_pred, n_clusters=n_clusters)
    return {f"{prefix}/tica-k-jsd": jsd}


def torus_kmeans_jsd(true_samples, pred_samples, topology, n_clusters=20, prefix=""):
    """JSD on KMeans-discretized dihedral angles."""
    true_traj = md.Trajectory(true_samples.cpu().numpy(), topology=topology)
    pred_traj = md.Trajectory(pred_samples.cpu().numpy(), topology=topology)

    dihedrals_true = _compute_dihedrals(true_traj)
    dihedrals_pred = _compute_dihedrals(pred_traj)

    jsd = _kmeans_jsd(dihedrals_true, dihedrals_pred, n_clusters=n_clusters)
    return {f"{prefix}/torus-k-jsd": jsd}
