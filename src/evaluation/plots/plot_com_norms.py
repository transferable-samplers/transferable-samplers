import logging

import matplotlib
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_com_norms(
    log_image_fn,
    proposal_samples,
    resampled_samples,
    smc_samples,
    ylim=None,
    prefix="",
    wandb_logger: WandbLogger = None,
):
    logging.info(f"Plotting com norms for {prefix}")
    if proposal_samples is not None:
        proposal_samples_com_norm = proposal_samples.mean(dim=1).norm(dim=-1).cpu()

    if resampled_samples is not None:
        resampled_samples_com_norm = resampled_samples.mean(dim=1).norm(dim=-1).cpu()

    if smc_samples is not None:
        smc_samples_com_norm = smc_samples.mean(dim=1).norm(dim=-1).cpu()

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")

    if proposal_samples is not None:
        ax.hist(
            proposal_samples_com_norm,
            density=True,
            alpha=0.4,
            color="r",
            histtype="step",
            linewidth=3,
            label="Proposal",
        )
    if resampled_samples is not None:
        ax.hist(
            resampled_samples_com_norm,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="b",
            label="Proposal (reweighted)",
        )
    if smc_samples is not None:
        ax.hist(
            smc_samples_com_norm,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="orange",
            label="SMC",
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xlabel(r"\|C\|", labelpad=-2)
    plt.ylabel("Normalized Density")
    plt.legend()

    fig.canvas.draw()

    log_image_fn(fig, f"{prefix}/com_norms")
    plt.close()
