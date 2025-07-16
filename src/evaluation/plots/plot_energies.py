import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_energies(
    log_image_fn,
    test_samples_energy,
    proposal_samples_energy,
    resampled_samples_energy,
    smc_samples_energy,
    max_energy=100,
    ylim=None,
    prefix="",
):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")

    if test_samples_energy is not None:
        test_samples_energy = test_samples_energy.cpu()
    if proposal_samples_energy is not None:
        proposal_samples_energy = proposal_samples_energy.cpu()
    if resampled_samples_energy is not None:
        resampled_samples_energy = resampled_samples_energy.cpu()
    if smc_samples_energy is not None:
        smc_samples_energy = smc_samples_energy.cpu()

    x_max = -float("inf")
    if max_energy is None:
        if test_samples_energy is not None:
            x_max = max(x_max, test_samples_energy.max())
        if proposal_samples_energy is not None:
            x_max = max(x_max, proposal_samples_energy.max())
        if resampled_samples_energy is not None:
            x_max = max(x_max, resampled_samples_energy.max())
        if smc_samples_energy is not None:
            x_max = max(x_max, smc_samples_energy.max())
    else:
        x_max = max_energy

    energy_cropper = (lambda x: torch.clamp(x, max=x_max - 0.1)) if x_max is not None else (lambda x: x)

    x_min = float("inf")
    if test_samples_energy is not None:
        x_min = min(x_min, test_samples_energy.min())
    if proposal_samples_energy is not None:
        x_min = min(x_min, proposal_samples_energy.min())
    if resampled_samples_energy is not None:
        x_min = min(x_min, resampled_samples_energy.min())
    if smc_samples_energy is not None:
        x_min = min(x_min, smc_samples_energy.min())

    bin_edges = np.linspace(x_min, x_max, 100)

    ax.hist(
        energy_cropper(test_samples_energy.cpu()),
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="g",
        histtype="step",
        linewidth=3,
        label="True data",
    )
    if proposal_samples_energy is not None:
        ax.hist(
            energy_cropper(proposal_samples_energy.cpu()),
            bins=bin_edges,
            density=True,
            alpha=0.4,
            color="r",
            histtype="step",
            linewidth=3,
            label="Proposal",
        )
    if resampled_samples_energy is not None:
        ax.hist(
            energy_cropper(resampled_samples_energy.cpu()),
            bins=bin_edges,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="b",
            label="Proposal (reweighted)",
        )
    if smc_samples_energy is not None:
        ax.hist(
            energy_cropper(smc_samples_energy.cpu()),
            bins=bin_edges,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="orange",
            label="SMC",
        )

    xticks = list(ax.get_xticks())
    xticks = xticks[1:-1]
    if max_energy is not None:
        new_tick = bin_edges[-1]
        custom_label = rf"$\geq {new_tick}$"
        xticks.append(new_tick)
        xtick_labels = [str(int(tick)) if tick != new_tick else custom_label for tick in xticks]
    else:
        xtick_labels = [str(int(tick)) for tick in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xlabel(r"$\mathcal{E}(x)$", labelpad=-5)  # , fontsize=35)
    plt.ylabel("Normalized Density")  # , fontsize=35)
    plt.legend()  # fontsize=30)

    fig.canvas.draw()

    log_image_fn(fig, f"{prefix}energies")
    plt.close()
