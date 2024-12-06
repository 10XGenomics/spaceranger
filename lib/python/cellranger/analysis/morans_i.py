#
# Copyright (c) 2020 10X Genomics, Inc. All rights reserved.
#
"""Moran's I calculation for Spatial Enrichment analysis.

Code derived from pysal/esda Moran's I implementation https://github.com/pysal/esda/blob/master/esda/moran.py.
Pysal/esda is licensed under the BSD 3-Clause license (https://github.com/pysal/esda/blob/master/LICENSE)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp_sparse
import scipy.spatial.distance as sp_dist
import scipy.stats as sp_stats
import sklearn.neighbors as sk_neighbors

import cellranger.analysis.stats as analysis_stats

MORANS_I_PERMUTATIONS = 0
KNN_NEIGHBORS = 36  # approximately 3 layers surrounding a spot


def get_neighbors_weight_matrix(matrix, use_neighbors=True, use_distance=True):
    """Returns a neighborhood graph of distance between points in the x,y matrix using a decay.

    function by distance.

    Args:
        matrix: A 2D matrix with X,Y positions of each element in its rows.
        use_neighbors: use all pairwise distances if False or use a certain number of neighbors if True
        use_distance: use 0/1 for neighbors if False or use a 1/distance^2 weight decay if True
    Returns:
        A sparse weight matrix
    """
    distance_mode = "distance" if use_distance else "connectivity"
    if use_neighbors:
        weight_matrix = sk_neighbors.kneighbors_graph(
            matrix,
            n_neighbors=min(KNN_NEIGHBORS, matrix.shape[0] - 1),
            mode=distance_mode,
        )
    else:
        weight_matrix = sp_sparse.csr_matrix(sp_dist.squareform(sp_dist.pdist(matrix, "euclidean")))
    weight_matrix.data **= 2
    weight_matrix.data = np.reciprocal(weight_matrix.data)  # weight decay 1/distance^2
    return weight_matrix


def util_calc(z, weight_matrix, weight_matrix_sum=None):
    """Util calculation for Moran's I.

    Given (spot, ) normalized gene expression vector z and (spot, spot) distance-scaled weight matrix weight_matrix
    Returns Moran's I value for that gene
    """
    if weight_matrix_sum is None:
        weight_matrix_sum = weight_matrix.sum()
    z_lag = weight_matrix * z  # slag(self.w, z)
    inum = (z * z_lag).sum()
    return len(z) / weight_matrix_sum * inum / np.dot(z.T, z)


def calculate_morans_i(y, weight_matrix, two_tailed=True):
    """Per gene Moran's I calculation and p-value calculation.

    Given (spot, ) gene expression vector y and (spot, spot) distance-scaled weight matrix weight_matrix

    Returns:
        Moran's I value for that gene and p-values based on randomization and permutations (if turned on).
    """
    y = np.asarray(y).flatten()
    y = y - y.mean()
    n = len(y)
    term_s0 = weight_matrix.sum()
    term_s1 = ((weight_matrix + weight_matrix.transpose(copy=True)).data ** 2).sum() / 2.0
    term_s2 = (
        np.array(weight_matrix.sum(axis=1).flatten() + weight_matrix.sum(axis=0).flatten()) ** 2
    ).sum()

    # variance under "randomization"
    # null hypothesis: counts are independent across spots; not necessarily Gaussian
    term_a = n * ((n * n - 3 * n + 3) * term_s1 - n * term_s2 + 3 * (term_s0 * term_s0))
    term_b = (
        ((y**4).sum() / n)
        / (((y**2).sum() / n) ** 2)
        * ((n * n - n) * term_s1 - 2 * n * term_s2 + 6 * (term_s0 * term_s0))
    )
    var_i_rand = (term_a - term_b) / ((n - 1) * (n - 2) * (n - 3) * (term_s0 * term_s0)) - (
        -1.0 / (n - 1)
    ) * (-1.0 / (n - 1))
    morans_i = util_calc(y, weight_matrix)
    z_rand = (morans_i - (-1.0 / (n - 1))) / (var_i_rand ** (1 / 2.0))
    if z_rand > 0:
        p_rand = 1 - sp_stats.norm.cdf(z_rand)
    else:
        p_rand = sp_stats.norm.cdf(z_rand)
    if two_tailed:
        p_rand *= 2.0
    # simulations
    if MORANS_I_PERMUTATIONS > 0:
        sim = np.fromiter(
            (
                util_calc(np.random.permutation(y), weight_matrix, term_s0)
                for _ in range(MORANS_I_PERMUTATIONS)
            ),
            float,
            count=MORANS_I_PERMUTATIONS,
        )
        # number of permutations where the magnitude of Moran's I was
        # higher than the original observed value
        num_greater = np.sum(np.abs(sim) >= np.abs(morans_i))

        # Two tailed p-value
        p_sim = (num_greater + 1) / MORANS_I_PERMUTATIONS
        p_sim = np.minimum(1, p_sim)

        return [morans_i, p_rand, p_sim]
    else:
        return [morans_i, p_rand]


def normalize_morani_matrix(matrix):
    """Matrix Normalization function for matrix input to Moran's I calc."""
    matrix.tocsc()

    m = analysis_stats.normalize_by_umi(matrix)

    # Use log counts
    m.data = np.log2(1 + m.data)
    return m
