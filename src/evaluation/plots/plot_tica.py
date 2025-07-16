import logging
import pickle

import matplotlib
import matplotlib.pyplot as plt
import mdtraj as md
from matplotlib.colors import LogNorm

from src.data.preprocessing.tica import tica_features, tica_features_ca

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_tic01(ax, tics, tics_lims, cmap="viridis"):
    _ = ax.hist2d(tics[:, 0], tics[:, 1], bins=100, norm=LogNorm(), cmap=cmap, rasterized=True)
    ax.set_xlabel("TIC0", fontsize=45)
    ax.set_ylabel("TIC1", fontsize=45)
    ax.set_ylim(tics_lims[:, 1].min(), tics_lims[:, 1].max())
    ax.set_xlim(tics_lims[:, 0].min(), tics_lims[:, 0].max())
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_tica(log_image_fn, samples, topology, tica_model, prefix=""):
    pred_traj_samples = md.Trajectory(samples.cpu().numpy(), topology=topology)

    if topology.n_residues > 4:
        ca_list = [atom.index for atom in topology.atoms if atom.name == "CA"]
        features = tica_features_ca(pred_traj_samples, ca_list=ca_list)
    else:
        features = tica_features(pred_traj_samples)

    tics = tica_model.transform(features)
    fig, ax = plt.subplots()
    ax = plot_tic01(ax, tics, tics_lims=tics)
    log_image_fn(fig, f"{prefix}/tica/plot")
    plt.close()
