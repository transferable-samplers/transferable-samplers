import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.evaluation.metrics.ramachandran import get_phi_psi_vectors

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_ramachandran(log_image_fn, samples, topology, prefix: str = ""):
    logging.info(f"Plotting Ramachandran for {prefix}")
    prefix += "/rama"

    phis, psis = get_phi_psi_vectors(samples, topology)

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
            ],
        )
        ax.set_xlabel(r"$\varphi$", fontsize=45)
        # ax.set_title("Boltzmann Generator", fontsize=45)
        ax.set_ylabel(r"$\psi$", fontsize=45)
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_ticks([])
        cbar = fig.colorbar(im, ticks=ticks)
        # cbar.ax.set_yticklabels(np.abs(-np.log(ticks/h.max())), fontsize=25)
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
        log_image_fn(fig, f"{prefix}/ramachandran_simple/{i}")
        plt.close()
