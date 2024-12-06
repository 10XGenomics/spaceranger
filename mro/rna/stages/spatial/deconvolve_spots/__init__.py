# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
"""Deconvolve spots.

Using the method based on https://www.nature.com/articles/s41467-022-30033-z
"""
from __future__ import annotations

import csv
import os

import h5py
import martian
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from sklearn.decomposition import LatentDirichletAllocation as fitLDA

import cellranger.matrix as cr_matrix
from cellranger.rna.library import GENE_EXPRESSION_LIBRARY_TYPE
from cellranger.spatial.deconvolution import get_lda_feature_counts

__MRO__ = """
stage DECONVOLVE_SPOTS(
    in  h5   filtered_matrix,
    in  path analysis,
    out path deconvolution,
    src py   "stages/spatial/deconvolve_spots",
)
"""
# Setup basic boundaries around input data
## use graph clusters initially to set the K
## minimum nubmer of features detected per spot
MIN_FEATURES = 100
## minimum number of UMI to keep a feature
MIN_UMI = 10


def split(args):
    if not args.analysis or not args.filtered_matrix:
        chunk_def = {}
    else:
        mem_gb = (
            int(cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(args.filtered_matrix)) + 1
        )  # We want to ceil, while int() floors. Adding 1 handles this unless the estimated size is integral,
        # which is a very rare case - and in which case we just use 1 GB of extra memory
        chunk_def = {"__mem_gb": mem_gb, "__vmem_gb": mem_gb * 2}
    return {"chunks": [], "join": chunk_def}


def join(
    args, outs, _chunk_def, _chunk_outs
):  # pylint: disable=too-many-locals,too-many-statements
    """Performs reference free spot deconvolution based on the method in https://www.nature.com/articles/s41467-022-30033-z.

    Args:
        filtered_matrix (str): path to filtered matrix
        analysis (str): path to analysis directory with contains analysis.h5

    Outs:
        deconvolution: directory in outs
        deconvolved_spots: csv for each k
        deconvolution_topics_features: json for each k

    """
    # Skip if no clustering or filtered matrix
    if not args.analysis or not args.filtered_matrix:
        outs.deconvolution = None
        return

    # Get number of graph clusters
    analysis_h5 = h5py.File(os.path.join(args.analysis, "analysis.h5"), "r")
    ## Skip if gene expression clustering not found
    if "/clustering/_gene_expression_graphclust/clusters" not in analysis_h5:
        outs.deconvolution = None
        return
    max_clusters = int(max(analysis_h5["/clustering/_gene_expression_graphclust/clusters"])) + 2

    # Run LDA based deconvolution
    gex_matrix = cr_matrix.CountMatrix.load_h5_file(args.filtered_matrix).select_features_by_type(
        GENE_EXPRESSION_LIBRARY_TYPE
    )
    # Select features with at least MIN_UMI and barcodes with at least MIN_FEATURES
    ## Most samples should not have any barcodes removed but in some cases this will happen.
    ## If barcodes are removed need to account for that when plotting.

    # TODO: consider not removing any barcodes. Probably won't make the LDA model very different
    subset_gex_matrix = gex_matrix.select_axis_above_threshold(
        axis=1, threshold=MIN_FEATURES
    ).select_axis_above_threshold(axis=0, threshold=MIN_UMI)

    total_num_bcs = subset_gex_matrix.bcs_dim
    ## select for features that are in < to 100% spots but > than 5% of spots
    numbcs_per_feature = subset_gex_matrix.get_numbcs_per_feature()
    idx_top_features = np.where(
        (numbcs_per_feature < total_num_bcs) & (numbcs_per_feature > total_num_bcs * 0.05)
    )[0]
    subset_gex_matrix = subset_gex_matrix.select_features(idx_top_features)

    ## Get feature IDs so they can be used to subset in the final model execution
    ids_top_features = subset_gex_matrix.feature_ref.get_feature_ids_by_type(
        GENE_EXPRESSION_LIBRARY_TYPE
    )

    ## if subsetting removes all barcodes and/or features don't return a deconvolved_spots.csv or deconvolution_topics_features.json
    if 0 in (subset_gex_matrix.m).shape:
        outs.deconvolution = None
        return

    # Fit LDA model on subsetted matrix
    model = fitLDA(n_components=max_clusters, random_state=0)
    model.fit(subset_gex_matrix.m.T)
    # unordered_prop is the proportions (deconvolution) of each topic in each spot
    # only using the features that went into the model but all barcodes
    # The clusters here are not in the final order of clusters
    unordered_prop = model.transform(gex_matrix.select_features_by_ids(ids_top_features).m.T)

    feature_names = subset_gex_matrix.feature_ref.get_feature_names()
    feature_ids = subset_gex_matrix.feature_ref.get_feature_ids_by_type(
        GENE_EXPRESSION_LIBRARY_TYPE
    )
    feature_ids = [x.decode("utf-8") for x in feature_ids]

    # get top features for each topic (right now using all features).
    # You can also choose whether to use normalized or unnormalized counts from the model
    top_features_out_of_order = get_lda_feature_counts(
        model=model,
        feature_names=feature_names,
        n_features=len(feature_names),
        normalized=True,
    )

    # Getting topics and down clustering. Not in the order we ultimately want
    unordered_topics = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

    # Reorder topics so that everything is in order
    link_topics, top_features, prop = reorder_lda_topics(
        unordered_topics, top_features_out_of_order, unordered_prop
    )

    # Generate the distance thresholds for getting a required number of clusters
    # dist_thresholds[i] is the threshold used to get i+2 clusters
    dist_thresholds = list(
        reversed(
            [0.5 * (link_topics[i][2] + link_topics[i + 1][2]) for i in range(max_clusters - 2)]
        )
    )

    min_cluster = 2
    os.makedirs(outs.deconvolution, exist_ok=True)

    for num_clusters in range(min_cluster, max_clusters + 1):
        if num_clusters == max_clusters:
            down_clustered_prop = prop
            num_down_clusters = max_clusters
            topic_clusters = np.arange(1, max_clusters + 1)
            top_features_downclustered = get_topic_features_and_log2fc(
                top_features, num_down_clusters, topic_clusters
            )

            # save the dendrogram using distances
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            hierarchy.dendrogram(
                link_topics, labels=range(1, max_clusters + 1), color_threshold=-1, ax=ax
            )
            plt.xlabel("Topic Number")
            plt.ylabel("Manhattan Distance")
            plt.savefig(
                os.path.join(outs.deconvolution, f"dendrogram_k{max_clusters}_distances.png"),
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
                os.path.join(outs.deconvolution, f"dendrogram_k{max_clusters}.png"),
                format="png",
                bbox_inches="tight",
            )
            plt.close()

        else:
            # Get the distance threshold to use
            dist_threshold_to_use = dist_thresholds[num_clusters - 2]
            # Get num_clusters from topic clusters
            out_of_order_topic_clusters = hierarchy.fcluster(
                link_topics, dist_threshold_to_use, criterion="distance"
            )
            topic_clusters = reorder_topics(out_of_order_topic_clusters)
            # handles the case when heirarchical clustering returns less than maxclust
            num_down_clusters = max(topic_clusters)

            # Generate an incidence matrix of downsampled clusters with
            topic_to_down_cluster_matrix = np.zeros((max_clusters, num_down_clusters))
            for index, topic_cluster in enumerate(topic_clusters):
                topic_to_down_cluster_matrix[index, topic_cluster - 1] = 1
            # Generate loadings of each barcode. This is the sum of loadings
            # on the underlying clusters
            down_clustered_prop = prop.dot(topic_to_down_cluster_matrix)

            # Generate topics of the down clustering. These are the mean of the
            # topics of the underlying clusters
            top_features_downclustered = get_topic_features_and_log2fc(
                top_features, num_down_clusters, topic_clusters
            )

        t_matrix = np.matrix(top_features_downclustered).T.astype(str)

        # Write out the CSV of deconvolved spots
        ## Make the directory for the specific k
        k_decon_dir = os.path.join(outs.deconvolution, f"deconvolution_k{num_clusters}")
        os.mkdir(k_decon_dir)

        csv_decon_path = martian.make_path(
            os.path.join(k_decon_dir, f"deconvolved_spots_k{num_clusters}.csv")
        ).decode()
        topic_names = get_cluster_names(topic_clusters)

        with open(csv_decon_path, "w") as deconvolved_spots_csv:
            writer = csv.writer(deconvolved_spots_csv)
            writer.writerow(
                ["Barcode"] + [f"Topic {topic_names[i]}" for i in range(1, num_down_clusters + 1)]
            )
            with np.printoptions(precision=6, suppress=True):
                for row in zip(gex_matrix.bcs.astype(str), down_clustered_prop):
                    writer.writerow([row[0]] + np.array_str(row[1]).strip("[]").split())
        # Write csv.gz of top features in each topic
        csv_features_path = martian.make_path(
            os.path.join(k_decon_dir, f"deconvolution_topic_features_k{num_clusters}.csv")
        ).decode()

        max_feature_len = max(len(x) for x in feature_names + feature_ids)
        np.savetxt(
            fname=csv_features_path,
            X=np.insert(
                t_matrix.astype(f"<U{max_feature_len}"), 0, [feature_ids, feature_names], axis=1
            ),
            delimiter=",",
            comments="",
            fmt="%s",
            header="Feature ID,Feature Name,"
            + ",".join(
                f"Feature count topic {topic_names[i]},Feature log2 fold change topic {topic_names[i]}"
                for i in range(1, num_down_clusters + 1)
            ),
        )


def reorder_lda_topics(
    unordered_topics: np.ndarray,
    top_features_out_of_order: dict[int, list[tuple[str, float]]],
    unordered_prop: np.ndarray,
) -> tuple[np.ndarray, dict[int, list[tuple[str, float]]], np.ndarray]:
    """Rename and reorder topics learnt.

    so that the hierarchical clustering is in
    order and dendograms are in order.

    Args:
        unordered_topics (np.ndarray): Unordered topics (this is a (k x n)-matrix with each row
            summing to 1.0) - n: number of features, k: number of topics
        top_features_out_of_order (dict[int, tuple[str, float]]): dictionary of the following form
            {0: [("Richard Woods", 8.9), ("Rafael Gonzalez", 9.0), ...],
             1: [...],
             ...
            }
        unordered_prop (np.ndarray): Proportions of an unordered topics in barcodes (this is a
            (m x k)-matrix with each row summing to 1.0) - m: number of barcodes

    Returns:
        tuple[np.ndarray, dict[int, tuple[str, float]], np.ndarray]: (link_topics, top_features, prop)
            link_topics: a ((k-1) x 4) linkage matrix of hierarchical clustering of topics
            top_features: reordered top_features_out_of_order  dict
            prop:  Proportions of an unordered topics in barcodes (this is a
                (m x k)-matrix with each row summing to 1.0)
    """
    # Running hierarchical clustering on the topics. Uses average linkage with TV distance
    # First a pass to reorder the topics.
    link_topics_first_pass = hierarchy.linkage(unordered_topics, "average", "cityblock")
    # Then reordering topics so that the dendrogram renders in order
    # Both in the topic names and in the proportions
    order_of_topics = hierarchy.leaves_list(link_topics_first_pass)

    # Reorder topics and the proportions
    topics = unordered_topics[order_of_topics, :]
    prop = unordered_prop[:, order_of_topics]
    link_topics = hierarchy.linkage(topics, "average", "cityblock")

    # Reorder features to be in right order
    # As order_of_topics is a permutation of 0:max_clusters, argsort(order_of_topics)
    # is also a permutation of 0:max_clusters with argsort(order_of_topics)[a] = b
    # iff order_of_topics[b] = a. Thus argsort(order_of_topics) is the
    # "inverse image" of order_of_topics.
    invert_order_of_topics = np.argsort(order_of_topics)
    top_features = {
        invert_order_of_topics[key]: val for key, val in top_features_out_of_order.items()
    }

    return link_topics, top_features, prop


def get_cluster_names(topic_clusters: np.ndarray) -> dict[int, str]:
    """Takes in topics and reconstructs topic names from them.

    The topics are expected to be in sorted order as returned by
    `reorder_topics` as we expect the merged topics to be consecutive integers.
    For example topic_clusters np.array([1, 2, 2, 2, 3])
    gives us the names_dict {1: '1', 2: '2to4', 3: '5'}.

    Args:
        topic_clusters (np.ndarray): sorted topics in

    Returns:
        dict: Dict of topic_number: name
    """
    cluster_to_base_clusters = {}
    for ind, x in enumerate(topic_clusters):
        cluster_to_base_clusters[x] = cluster_to_base_clusters.get(x, []) + [ind + 1]

    cluster_to_base_clusters_pruned = {
        x: [str(min(lst)), str(max(lst))] if len(lst) > 1 else [str(lst[0])]
        for x, lst in cluster_to_base_clusters.items()
    }

    names_dict = {name: "to".join(lst) for name, lst in cluster_to_base_clusters_pruned.items()}
    return names_dict


def reorder_topics(topic_clusters: np.ndarray) -> np.ndarray:
    """Takes an order of topics in an unsorted order and sorts them.

    for example np.array([13, 12,  1,  2,  3,  4,  5,  6, 11,  8,  7,  7,  9, 10])
    gets sorted to np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13])
    The topics get renumbered so that they are in order.

    Args:
        topic_clusters (np.ndarray): raw topics in

    Returns:
        np.ndarray: sorted topics out
    """
    order_dict = {}
    next_value = 1
    for val in topic_clusters:
        if val not in order_dict:
            order_dict[val] = next_value
            next_value += 1
    return np.array([order_dict[x] for x in topic_clusters])


def get_topic_features_and_log2fc(
    top_features: dict, num_down_clusters: int, topic_clusters: np.ndarray
) -> list[np.ndarray]:
    """Generate topics of the down clustering.

    These are the mean of the topics of the underlying clusters.

    Args:
        top_features (Dict): Dict of top features and their expression for clusters in same sorted order for each cluster
        num_down_clusters (int): number of clusters
        topic_clusters (np.ndarray): a numpy array indicating where each cluster is collapsed

    Returns:
        list[np.ndarray]: list of numpy arrays with feature values and fold change
    """
    top_features_downclustered = []
    for cluster_number in range(num_down_clusters):
        clusters_collapsed = np.where(topic_clusters == cluster_number + 1)[0]
        background_clusters = np.where(topic_clusters != cluster_number + 1)[0]

        feature_values = np.mean(
            [[x[1] for x in top_features[y]] for y in clusters_collapsed], axis=0
        )
        background_values = np.mean(
            [[x[1] for x in top_features[y]] for y in background_clusters], axis=0
        )
        fold_change = np.nan_to_num((feature_values / background_values), nan=1)

        top_features_downclustered.append(feature_values.astype(int).astype(str))
        top_features_downclustered.append(np.log2(fold_change).astype("|S7").astype("str"))
    return top_features_downclustered
