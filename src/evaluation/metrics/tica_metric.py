import pickle

import deeptime as dt
import mdtraj as md
import numpy as np
import torch

from src.data.preprocessing.tica import tica_features, tica_features_ca
from src.evaluation.metrics.distribution_distances import distribution_distances

def tica_metric(true_samples, pred_samples, topology, tica_model, prefix=""):

    true_traj_samples = md.Trajectory(true_samples.cpu().numpy(), topology=topology)
    pred_traj_samples = md.Trajectory(pred_samples.cpu().numpy(), topology=topology)

    if topology.n_residues > 4:
        ca_list = [atom.index for atom in topology.atoms if atom.name == "CA"]
        features_test = tica_features_ca(true_traj_samples, ca_list=ca_list)
        features = tica_features_ca(pred_traj_samples, ca_list=ca_list)
    else:
        features_test = tica_features(true_traj_samples)
        features = tica_features(pred_traj_samples)

    n = min(len(features_test), len(features))
    tics_test = torch.Tensor(tica_model.transform(features_test))[:n, 0:2]
    tics = torch.Tensor(tica_model.transform(features))[:n, 0:2]
    return distribution_distances(tics_test, tics, prefix=prefix + "/tica")
