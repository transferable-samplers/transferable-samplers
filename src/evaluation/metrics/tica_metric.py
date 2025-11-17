import pickle

import mdtraj as md
import torch

from src.data.preprocessing.tica import tica_features
from src.evaluation.metrics.distribution_distances import distribution_distances


def tica_metric(true_samples, pred_samples, topology, tica_model=None, tica_model_path=None, prefix=""):
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
    return distribution_distances(tics_test, tics, prefix=prefix + "/tica")
