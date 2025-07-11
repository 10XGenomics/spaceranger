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

from cellranger.spatial.image import base64_encode_image


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


def heatmap_legend(cmap, title, vmin, vmax, fname):
    """Generate a heatmap legend image for the UMI plot."""
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colorbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    colorbar.ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=100, pad_inches=0)
    plt.close()
    return base64_encode_image(fname)


def get_cmap(cmap_name):
    """Get a cmap from name."""
    return plt.get_cmap(cmap_name)


COLOR_PALETTE_NAME = "turbo"


def rgb_to_hex(rgb: list[int]) -> str:
    """Convert an RGB color to a hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _color_map_rgb(norm_value: float) -> list[int]:
    """Get the color for a value in a color map."""
    color_map = plt.get_cmap(COLOR_PALETTE_NAME)
    return color_map(norm_value, bytes=True)[:3]


def calculate_cluster_position_in_colormap(cluster_num, num_clusters) -> int:
    """Sample odd clusters from the left side of the color map and even clusters from the right side.

    Arguments:
        cluster_num: cluster number (1-indexed).
        num_clusters: total number of clusters.

    Returns:
        int: position of the cluster in the color map. (0-indexed)
    """
    assert 1 <= cluster_num <= num_clusters
    cluster_num = cluster_num - 1  # 0-indexed
    if cluster_num % 2 == 0:
        return cluster_num // 2
    else:
        half_way = (num_clusters + 1) // 2
        return half_way + cluster_num // 2


def cluster_color_rgb(cluster_num: int, num_clusters: int) -> list[int]:
    """Get the color for a cluster. Cluster numbers start at 1."""
    if num_clusters == 1:
        return _color_map_rgb(0.5)

    cluster_pos = calculate_cluster_position_in_colormap(cluster_num, num_clusters)

    return _color_map_rgb(cluster_pos / (num_clusters - 1))
