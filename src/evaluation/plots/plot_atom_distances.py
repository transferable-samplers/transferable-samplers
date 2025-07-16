import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def interatomic_dist(x, flatten=True):
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
    log_image_fn,
    true_samples,
    proposal_samples,
    resampled_samples,
    smc_samples,
    ylim=None,
    prefix="",
    wandb_logger: WandbLogger = None,
):
    logging.info(f"Plotting interatomic distances for {prefix}")
    true_samples_dist = interatomic_dist(true_samples).cpu()
    min_dist = true_samples_dist.min()
    max_dist = true_samples_dist.max()

    if proposal_samples is not None:
        proposal_samples_dist = interatomic_dist(proposal_samples).cpu()
        min_dist = min(min_dist, proposal_samples_dist.min())
        max_dist = max(max_dist, proposal_samples_dist.max())

    if resampled_samples is not None:
        resampled_samples_dist = interatomic_dist(resampled_samples).cpu()
        min_dist = min(min_dist, resampled_samples_dist.min())
        max_dist = max(max_dist, resampled_samples_dist.max())

    if smc_samples is not None:
        smc_samples_dist = interatomic_dist(smc_samples).cpu()
        min_dist = min(min_dist, smc_samples_dist.min())
        max_dist = max(max_dist, smc_samples_dist.max())

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
    if proposal_samples is not None:
        ax.hist(
            proposal_samples_dist,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            color="r",
            histtype="step",
            linewidth=3,
            label="Proposal",
        )
    if resampled_samples is not None:
        ax.hist(
            resampled_samples_dist,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="b",
            label="Proposal (reweighted)",
        )
    if smc_samples is not None:
        ax.hist(
            smc_samples_dist,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="orange",
            label="SMC",
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xlabel("Interatomic Distance  ", labelpad=-2)  # , fontsize=35)
    plt.ylabel("Normalized Density")  # , fontsize=35)
    plt.legend()  # fontsize=30)

    fig.canvas.draw()

    log_image_fn(fig, f"{prefix}/interatomic_distances")
    plt.close()
