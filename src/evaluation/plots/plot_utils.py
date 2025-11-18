"""Shared utility functions for plotting."""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_histogram_comparison(
    ax,
    data_dict,
    bin_edges=None,
    color_map=None,
    label_map=None,
):
    """
    Plot multiple histograms on the same axis with consistent styling.

    Args:
        ax: Matplotlib axis object to plot on
        data_dict: Dictionary mapping data keys to data arrays
        bin_edges: Bin edges for the histogram (None for auto bins)
        color_map: Optional dictionary mapping data keys to colors
        label_map: Optional dictionary mapping data keys to labels
    """
    default_colors = {
        "true": "g",
        "proposal": "r",
        "resampled": "b",
        "smc": "orange",
    }
    default_labels = {
        "true": "True data",
        "proposal": "Proposal",
        "resampled": "Proposal (reweighted)",
        "smc": "SMC",
    }

    if color_map is None:
        color_map = default_colors
    if label_map is None:
        label_map = default_labels

    for key, data in data_dict.items():
        if data is not None:
            color = color_map.get(key, "black")
            label = label_map.get(key, key)
            hist_kwargs = {
                "density": True,
                "alpha": 0.4,
                "color": color,
                "histtype": "step",
                "linewidth": 3,
                "label": label,
            }
            if bin_edges is not None:
                hist_kwargs["bins"] = bin_edges
            ax.hist(data, **hist_kwargs)
