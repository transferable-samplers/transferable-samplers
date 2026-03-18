from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

COLORS = ["r", "b", "orange", "purple", "brown", "pink"]


def _interatomic_dist(x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """Compute upper-triangular pairwise interatomic distances."""
    assert len(x.shape) == 3, f"Expected 3D array, got {x.shape}"

    num_atoms = x.shape[1]

    # Compute the pairwise interatomic distances
    # removes duplicates and diagonal
    distances = x[:, None, :, :] - x[:, :, None, :]
    distances = distances[
        :,
        torch.triu(torch.ones((num_atoms, num_atoms)), diagonal=1) == 1,
    ]
    dist = torch.linalg.norm(distances, dim=-1)

    if flatten:
        dist = dist.flatten()
    return dist


def plot_atom_distances(
    log_image_fn: Callable[[Any, str], None],
    true_samples: torch.Tensor,
    samples_dict: dict[str, torch.Tensor],
    ylim: tuple[float, float] | None = None,
    prefix: str = "",
) -> None:
    """Plot interatomic distance histograms for ground truth and generated sample sets.

    Args:
        log_image_fn: Callable to log the figure.
        true_samples: Ground truth samples tensor [batch, atoms, 3].
        samples_dict: Dict mapping sample set names to sample tensors.
        ylim: Optional y-axis limits.
        prefix: Metric key prefix.
    """
    logging.info(f"Plotting interatomic distances for {prefix}")
    true_samples_dist = _interatomic_dist(true_samples).cpu()
    min_dist = true_samples_dist.min()
    max_dist = true_samples_dist.max()

    named_dists = {}
    for name, samples in samples_dict.items():
        dist = _interatomic_dist(samples).cpu()
        named_dists[name] = dist
        min_dist = min(min_dist, dist.min())
        max_dist = max(max_dist, dist.max())

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")
    bin_edges = np.linspace(min_dist, max_dist, 100)

    ax.hist(
        true_samples_dist,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="g",
        histtype="step",
        linewidth=3,
        label="True data",
    )

    for (name, dist), color in zip(named_dists.items(), COLORS, strict=False):
        ax.hist(
            dist,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            color=color,
            histtype="step",
            linewidth=3,
            label=name,
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xlabel("Interatomic Distance  ", labelpad=-2)
    plt.ylabel("Normalized Density")
    plt.legend()

    fig.canvas.draw()

    log_image_fn(fig, f"{prefix}/interatomic-distances")
    plt.close()
