import pickle

import mdtraj as md
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans

from src.data.preprocessing.tica import tica_features


def compute_hist(states, n_clusters=20):
    counts = np.bincount(states, minlength=n_clusters)

    # # Ensure no zero probabilities if a state is unvisited, which can cause issues in divergence calculations
    epsilon = 1e-10
    dist = (counts + epsilon) / (counts.sum() + n_clusters * epsilon)

    return dist


def jsd_metric(true_samples, pred_samples, topology, tica_model=None, tica_model_path=None, n_clusters=20, prefix=""):
    assert (tica_model is not None) ^ (tica_model_path is not None), "Provide either tica_model or tica_model_path."

    if not tica_model:
        with open(tica_model_path, "rb") as f:
            tica_model = pickle.load(f)  # noqa: S301

    true_traj_samples = md.Trajectory(true_samples.cpu().numpy(), topology=topology)
    pred_traj_samples = md.Trajectory(pred_samples.cpu().numpy(), topology=topology)

    features_test = tica_features(true_traj_samples)
    features = tica_features(pred_traj_samples)

    n = min(len(features_test), len(features))
    tics_test = torch.Tensor(tica_model.transform(features_test))[:n, 0:2]
    tics = torch.Tensor(tica_model.transform(features))[:n, 0:2]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(tics_test)
    true_states = kmeans.labels_  # Discretized trajectory
    gen_states = kmeans.predict(tics)  # Discretized generated samples

    true_dist = compute_hist(true_states, n_clusters=n_clusters)
    gen_dist = compute_hist(gen_states, n_clusters=n_clusters)

    jsd = jensenshannon(true_dist, gen_dist, base=2) ** 2
    return {
        f"{prefix}/jsd": jsd,
    }
