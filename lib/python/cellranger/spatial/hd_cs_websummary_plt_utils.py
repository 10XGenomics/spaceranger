# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Matplotlib functions used in the HD clustering plot."""

# isort: off
# pylint: disable=wrong-import-position
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pylint: enable=wrong-import-position
# isort: on


def plot_umap_image(umap_x, umap_y, umap_plot_fname, color_hex, min_xy, max_xy):
    """Plot HD UMAP image."""
    plt.figure(figsize=(4, 4), dpi=100, facecolor="none")
    plt.scatter(umap_x, umap_y, color=color_hex, s=0.1)
    plt.xlim([min_xy, max_xy])
    plt.ylim([min_xy, max_xy])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.savefig(umap_plot_fname, pad_inches=0, transparent=True)
    plt.close()


def get_cmap(cmap_name):
    """Get a cmap from name."""
    return plt.get_cmap(cmap_name)
