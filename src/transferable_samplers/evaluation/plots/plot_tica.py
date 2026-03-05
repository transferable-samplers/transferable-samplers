from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from transferable_samplers.data.preprocessing.tica import tica_features

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_tic01(ax: Axes, tics: np.ndarray, tics_lims: np.ndarray, cmap: str = "viridis") -> Axes:
    """Plot TIC0 vs TIC1 as a 2D histogram on the given axes."""
    _ = ax.hist2d(tics[:, 0], tics[:, 1], bins=100, norm=LogNorm(), cmap=cmap, rasterized=True)
    ax.set_xlabel("TIC0", fontsize=45)
    ax.set_ylabel("TIC1", fontsize=45)
    ax.set_ylim(tics_lims[:, 1].min(), tics_lims[:, 1].max())
    ax.set_xlim(tics_lims[:, 0].min(), tics_lims[:, 0].max())
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_tica(
    log_image_fn: Callable[[Any, str], None], samples: torch.Tensor, topology: Any, tica_model: Any, prefix: str = ""
) -> None:
    """Plot TICA projection (TIC0 vs TIC1) for a set of conformations.

    Args:
        log_image_fn: Callable to log the figure.
        samples: Conformation tensor ``(batch, num_atoms, 3)``.
        topology: mdtraj topology for feature computation.
        tica_model: Fitted TICA model for projection.
        prefix: Metric key prefix.
    """
    logging.info(f"Plotting TICA for {prefix}")

    pred_traj_samples = md.Trajectory(samples.cpu().numpy(), topology=topology)

    features = tica_features(pred_traj_samples)

    tics = tica_model.transform(features)
    if isinstance(tics, torch.Tensor):
        tics = tics.cpu().numpy()

    fig, ax = plt.subplots()
    ax = plot_tic01(ax, tics, tics_lims=tics)
    log_image_fn(fig, f"{prefix}/tica/plot")
    plt.close()
