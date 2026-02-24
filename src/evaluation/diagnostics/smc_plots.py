"""SMC diagnostic plots from trajectory snapshots and per-step diagnostics."""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm


def plot_smc_diagnostics(
    diagnostics_output: dict,
    log_image_fn: Callable,
    do_energy_plots: bool = False,
):
    """Plot all SMC diagnostics.

    Args:
        diagnostics_output: dict with keys:
            - "trajectory": list of SMCParticles snapshots (collected every log_traj_freq)
            - "diagnostics": dict with per-step lists: "t", "ess", "sigma", "acceptance_rate"
        log_image_fn: callable (fig, name) -> None for logging figures.
        do_energy_plots: whether to plot energy diagnostics from trajectory snapshots.
    """
    diagnostics = diagnostics_output["diagnostics"]
    trajectory = diagnostics_output["trajectory"]

    t_list = diagnostics["t"]
    ess_list = diagnostics["ess"]
    sigma_list = diagnostics["eps"]
    acceptance_rate_list = diagnostics["acceptance_rate"]

    # Lineage survival from trajectory snapshots
    if trajectory:
        survived_lineages = [
            p.lineage.unique().numel() / len(p) for p in trajectory
        ]
        traj_t = [t_list[0]] + [t_list[min(i, len(t_list) - 1)] for i in range(len(trajectory))]
        # Align: trajectory may have fewer points than diagnostics
        _plot_particle_survival(survived_lineages, log_image_fn)

    # Per-step scalar diagnostics
    _plot_weights_and_ess(trajectory, ess_list, t_list, log_image_fn)
    _plot_sigma(sigma_list, t_list, log_image_fn)
    _plot_acceptance_rate(acceptance_rate_list, t_list, log_image_fn)

    if do_energy_plots and trajectory:
        target_energy_list = [p.E_target.cpu().numpy() for p in trajectory]
        interpolation_energy_list = []
        for i, p in enumerate(trajectory):
            # Approximate t from trajectory position
            t_approx = i / max(len(trajectory) - 1, 1)
            interpolation_energy_list.append(
                ((1 - t_approx) * p.E_source.cpu() + t_approx * p.E_target.cpu()).numpy()
            )
        traj_t_list = [i / max(len(trajectory) - 1, 1) for i in range(len(trajectory))]
        _plot_stepwise_energy(target_energy_list, interpolation_energy_list, traj_t_list, log_image_fn)
        _plot_stepwise_energy_hist(target_energy_list, interpolation_energy_list, traj_t_list, log_image_fn)


def _plot_stepwise_energy(target_energy_list, interpolation_energy_list, t_list, log_image_fn):
    stepwise_target_energy_np = np.stack(target_energy_list)
    stepwise_interpolation_energy_np = np.stack(interpolation_energy_list)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for k in range(stepwise_target_energy_np.shape[1]):
        axs[0].plot(t_list, stepwise_target_energy_np[:, k], linewidth=1, alpha=0.5)
        axs[1].plot(t_list, stepwise_interpolation_energy_np[:, k], linewidth=1, alpha=0.5)

    axs[0].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Target energy", fontsize=12)
    axs[1].set_xlabel("Time", fontsize=12)
    axs[1].set_ylabel("Interpolation energy", fontsize=12)

    plt.tight_layout()
    log_image_fn(fig, "langevin/energies")
    plt.close()


def _plot_stepwise_energy_hist(target_energy_list, interpolation_energy_list, t_list, log_image_fn):
    stepwise_target_energy_np = np.stack(target_energy_list)
    stepwise_interpolation_energy_np = np.stack(interpolation_energy_list)
    t_np = np.array(t_list)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    data = stepwise_target_energy_np
    bins = np.linspace(data.min(), data.max(), 100)
    histograms = np.array([np.histogram(row, bins=bins)[0] for row in data])
    histograms_normalized = histograms / histograms.sum(axis=1, keepdims=True)
    extent = [t_np.min(), t_np.max(), bins[0], bins[-1]]
    im = axs[0].imshow(
        histograms_normalized.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        norm=LogNorm(
            vmin=histograms_normalized[histograms_normalized > 0].min(),
            vmax=histograms_normalized.max(),
        ),
        cmap="inferno",
    )
    axs[0].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Target energy", fontsize=12)
    fig.colorbar(im, ax=axs[0], label="Log Marginal Density")

    data = stepwise_interpolation_energy_np
    bins = np.linspace(data.min(), data.max(), 100)
    histograms = np.array([np.histogram(row, bins=bins)[0] for row in data])
    histograms_normalized = histograms / histograms.sum(axis=1, keepdims=True)
    extent = [t_np.min(), t_np.max(), bins[0], bins[-1]]
    im = axs[1].imshow(
        histograms_normalized.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        norm=LogNorm(
            vmin=histograms_normalized[histograms_normalized > 0].min(),
            vmax=histograms_normalized.max(),
        ),
        cmap="inferno",
    )
    axs[1].set_xlabel("Time", fontsize=12)
    axs[1].set_ylabel("Interpolation energy", fontsize=12)
    fig.colorbar(im, ax=axs[1], label="Log Marginal Density")

    plt.tight_layout()
    log_image_fn(fig, "langevin/energy_histograms")
    plt.close()


def _plot_weights_and_ess(trajectory, ess_list, t_list, log_image_fn):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot logw trajectories from snapshots if available
    if trajectory:
        logw_list = [p.logw for p in trajectory]
        A_np = torch.stack(logw_list).cpu().numpy()
        traj_t = np.linspace(0, 1, len(trajectory))
        for k in range(A_np.shape[1]):
            axs[0].plot(traj_t, A_np[:, k], linewidth=1, alpha=0.5)
    axs[0].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("log w", fontsize=12)

    axs[1].plot(t_list, ess_list, linewidth=1, alpha=0.3)
    axs[1].set_xlabel("Time", fontsize=12)
    axs[1].set_ylabel("ESS", fontsize=12)
    axs[1].set_yscale("log")

    plt.tight_layout()
    log_image_fn(fig, "langevin/weights")
    plt.close()


def _plot_sigma(sigma_list, t_list, log_image_fn):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.plot(t_list, sigma_list, linewidth=1, alpha=0.5)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Sigma", fontsize=12)
    plt.tight_layout()
    log_image_fn(fig, "langevin/sigma")
    plt.close()


def _plot_acceptance_rate(acceptance_rate_list, t_list, log_image_fn):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.plot(t_list, acceptance_rate_list, linewidth=1, alpha=0.5)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Acceptance Rate", fontsize=12)
    plt.tight_layout()
    log_image_fn(fig, "langevin/acceptance-rate")
    plt.close()


def _plot_particle_survival(survived_lineages, log_image_fn):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.plot(range(len(survived_lineages)), survived_lineages, linewidth=1, alpha=0.5)
    ax.set_xlabel("Snapshot", fontsize=12)
    ax.set_ylabel("Survived Lineages (%)", fontsize=12)
    plt.tight_layout()
    log_image_fn(fig, "langevin/lineage-survived")
    plt.close()
