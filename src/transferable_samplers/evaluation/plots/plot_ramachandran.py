from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import torch
from matplotlib.colors import LogNorm

from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def _get_paired_phi_psi_vectors(samples: torch.Tensor, topology: md.Topology) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-residue phi/psi dihedral angles, paired by the residue they belong to.

    ``md.compute_phi`` omits the first residue when there is no preceding atom (e.g. no
    ACE cap) to form the angle with, and ``md.compute_psi`` omits the last residue when
    there is no following atom (e.g. no NME cap). Without capping groups, the returned
    phi/psi columns are therefore offset by one residue relative to each other, so a
    Ramachandran plot needs residues matched explicitly via the atom indices mdtraj
    returns rather than zipped by column position. This holds regardless of whether
    capping groups are present.
    """
    samples = samples.cpu()
    traj = md.Trajectory(samples, topology=topology)
    phi_indices, phis = md.compute_phi(traj)
    psi_indices, psis = md.compute_psi(traj)

    # phi is C(i-1)-N(i)-CA(i)-C(i): the N atom (column 1) belongs to residue i.
    phi_residues = [topology.atom(idx).residue.index for idx in phi_indices[:, 1]]
    # psi is N(i)-CA(i)-C(i)-N(i+1): the N atom (column 0) belongs to residue i.
    psi_residues = [topology.atom(idx).residue.index for idx in psi_indices[:, 0]]

    common_residues = [r for r in phi_residues if r in psi_residues]
    phi_cols = [phi_residues.index(r) for r in common_residues]
    psi_cols = [psi_residues.index(r) for r in common_residues]

    return phis[:, phi_cols], psis[:, psi_cols]


def plot_ramachandran(
    log_image_fn: Callable[[Any, str], None], samples: torch.Tensor, topology: Any, prefix: str = ""
) -> None:
    """Plot per-residue Ramachandran (phi/psi) density maps.

    Generates two versions per residue: a free-energy-scaled plot and a
    simple count-based plot.

    Args:
        log_image_fn: Callable to log the figure.
        samples: Conformation tensor ``(batch, num_atoms, 3)``.
        topology: mdtraj topology for dihedral computation.
        prefix: Metric key prefix.
    """
    logger.info(f"Plotting Ramachandran for {prefix}")
    prefix += "/rama"

    phis, psis = _get_paired_phi_psi_vectors(samples, topology)

    for i in range(phis.shape[1]):
        phi_tmp = phis[:, i]
        psi_tmp = psis[:, i]
        fig, ax = plt.subplots()
        plot_range = [-np.pi, np.pi]
        h, x_bins, y_bins, im = ax.hist2d(
            phi_tmp,
            psi_tmp,
            100,
            norm=LogNorm(),
            range=[plot_range, plot_range],
            rasterized=True,
        )
        ticks = np.array(
            [
                np.exp(-6) * h.max(),
                np.exp(-4.0) * h.max(),
                np.exp(-2) * h.max(),
                h.max(),
            ]
        )
        ax.set_xlabel(r"$\varphi$", fontsize=45)
        # ax.set_title("Boltzmann Generator", fontsize=45)
        ax.set_ylabel(r"$\psi$", fontsize=45)
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_ticks([])
        cbar = fig.colorbar(im, ticks=ticks)
        # cbar.ax.set_yticklabels(np.abs(-np.log(ticks/h.max())), fontsize=25)
        # pyrefly: ignore [bad-argument-type]
        cbar.ax.set_yticklabels([6.0, 4.0, 2.0, 0.0], fontsize=25)

        cbar.ax.invert_yaxis()
        cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
        log_image_fn(fig, f"{prefix}/ramachandran/{i}")

        phi_tmp = phis[:, i]
        psi_tmp = psis[:, i]
        fig, ax = plt.subplots()
        plot_range = [-np.pi, np.pi]
        h, x_bins, y_bins, im = ax.hist2d(
            phi_tmp,
            psi_tmp,
            100,
            norm=LogNorm(),
            range=[plot_range, plot_range],
            rasterized=True,
        )
        ax.set_xlabel(r"$\varphi$", fontsize=45)
        ax.set_ylabel(r"$\psi$", fontsize=45)
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_ticks([])
        cbar = fig.colorbar(im)  # , ticks=ticks)
        im.set_clim(vmax=samples.shape[0] // 20)
        cbar.ax.set_ylabel(f"Count, max = {int(h.max())}", fontsize=18)
        log_image_fn(fig, f"{prefix}/ramachandran-simple/{i}")
        plt.close()
