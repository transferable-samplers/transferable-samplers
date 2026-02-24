import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

COLORS = ["r", "b", "orange", "purple", "brown", "pink"]


def plot_energies(
    log_image_fn,
    true_energy,
    samples_energy_dict: dict[str, torch.Tensor],
    max_energy=100,
    ylim=None,
    prefix="",
):
    """Plot energy histograms for ground truth and generated sample sets.

    Args:
        log_image_fn: Callable to log the figure.
        true_energy: Ground truth energy tensor.
        samples_energy_dict: Dict mapping sample set names to energy tensors.
        max_energy: Maximum energy for x-axis clipping.
        ylim: Optional y-axis limits.
        prefix: Metric key prefix.
    """
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")

    true_energy = true_energy.cpu()

    all_energies = [true_energy]
    for energy in samples_energy_dict.values():
        all_energies.append(energy.cpu())

    if max_energy is None:
        x_max = max(e.max() for e in all_energies)
    else:
        x_max = max_energy

    energy_cropper = (lambda x: torch.clamp(x, max=x_max - 0.1)) if x_max is not None else (lambda x: x)

    x_min = min(e.min() for e in all_energies)
    bin_edges = np.linspace(x_min, x_max, 100)

    ax.hist(
        energy_cropper(true_energy),
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="g",
        histtype="step",
        linewidth=3,
        label="True data",
    )

    for (name, energy), color in zip(samples_energy_dict.items(), COLORS, strict=False):
        ax.hist(
            energy_cropper(energy.cpu()),
            bins=bin_edges,
            density=True,
            alpha=0.4,
            color=color,
            histtype="step",
            linewidth=3,
            label=name,
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

    plt.xlabel(r"$\mathcal{E}(x)$", labelpad=-5)
    plt.ylabel("Normalized Density")
    plt.legend()

    fig.canvas.draw()

    log_image_fn(fig, f"{prefix}energies")
    plt.close()
