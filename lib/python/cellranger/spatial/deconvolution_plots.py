#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#
"""Plotting utilities for use in spatial deconvolution."""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy


def save_dendrogram_figure(link_topics, max_clusters, outs_directory, prefix=None):
    """Generate and save dendrogram image."""
    dendrogram_prefix = f"{prefix}_dendrogram" if prefix else "dendrogram"
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    hierarchy.dendrogram(link_topics, labels=range(1, max_clusters + 1), color_threshold=-1, ax=ax)
    plt.xlabel("Topic Number")
    plt.ylabel("Manhattan Distance")
    plt.savefig(
        os.path.join(outs_directory, f"{dendrogram_prefix}_k{max_clusters}_distances.png"),
        format="png",
        bbox_inches="tight",
    )
    plt.close()

    # save dendogram of levels
    plt.figure(figsize=(10, 10))
    # Getting a new link topic with distance being given by the level.
    # The third column of the link_topics is distances in ascending order
    # Resetting this to this level number gives us level as distance
    level_link_topics = link_topics.copy()
    level_link_topics[:, 2] = np.arange(1, max_clusters)
    ax = plt.gca()
    hierarchy.dendrogram(
        level_link_topics, labels=range(1, max_clusters + 1), color_threshold=-1, ax=ax
    )
    ax.set_yticks(
        np.arange(max_clusters - 1),
        (max_clusters - np.arange(max_clusters - 1)).astype("int"),
    )
    ax.grid(axis="y")
    plt.xlabel("Topic Number")
    plt.ylabel("K")
    plt.savefig(
        os.path.join(outs_directory, f"{dendrogram_prefix}_k{max_clusters}.png"),
        format="png",
        bbox_inches="tight",
    )
    plt.close()
