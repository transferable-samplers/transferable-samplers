from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

COLORS = ["r", "b", "orange", "purple", "brown", "pink"]


def plot_com_norms(
    log_image_fn: Callable[[Any, str], None],
    samples_dict: dict[str, torch.Tensor],
    ylim: tuple[float, float] | None = None,
    prefix: str = "",
) -> None:
    """Plot center-of-mass norm histograms for generated sample sets.

    Args:
        log_image_fn: Callable to log the figure.
        samples_dict: Dict mapping sample set names to sample tensors.
        ylim: Optional y-axis limits.
        prefix: Metric key prefix.
    """
    logging.info(f"Plotting com norms for {prefix}")

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")

    for (name, samples), color in zip(samples_dict.items(), COLORS, strict=False):
        com_norm = samples.mean(dim=1).norm(dim=-1).cpu()
        ax.hist(
            com_norm,
            density=True,
            alpha=0.4,
            color=color,
            histtype="step",
            linewidth=3,
            label=name,
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xlabel(r"\|C\|", labelpad=-2)
    plt.ylabel("Normalized Density")
    plt.legend()

    fig.canvas.draw()

    log_image_fn(fig, f"{prefix}/com_norms")
    plt.close()
