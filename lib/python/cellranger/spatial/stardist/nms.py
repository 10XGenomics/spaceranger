#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#

"""Functions related to non maximal supression."""

import math

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def ind_prob_thresh(prob, prob_thresh, b: int | tuple[int] | None = 2):
    """Apply a probability threshold and boundary exclusion to an input probability map.

    This function creates a boolean mask indicating pixels where the probability
    `prob` is greater than `prob_thresh`. Optionally, it can also exclude a
    boundary region of the image from this mask.

    Parameters
    ----------
    prob : ndarray
        The input probability map (N-dimensional).
    prob_thresh : float
        The probability threshold. Pixels with probability values strictly
        greater than this threshold will be considered.
    b : int or tuple of tuples of int, optional
        Defines the size of the boundary to exclude.
        If `b` is a scalar integer, a boundary of this size is excluded
        from all sides of all dimensions.
        If `b` is a tuple of tuples, it specifies the boundary size for
        each dimension. For a D-dimensional input `prob`, `b` should be
        a tuple of D tuples, where each inner tuple `(b_start, b_end)`
        specifies the number of pixels to exclude from the start and end
        of that dimension. If `b_start` or `b_end` is 0 or None, no
        exclusion is applied to that boundary.
        If `b` is None, no boundary exclusion is performed.
        By default, a boundary of size 2 is excluded from all sides of all dimensions.

    Returns:
    -------
    ind_thresh : ndarray (bool)
        A boolean array of the same shape as `prob`. It is True for pixels
        where the probability is greater than `prob_thresh` and which are
        not within the excluded boundary region (if `b` is specified).
    """
    if b is not None and np.isscalar(b):
        b = ((b, b),) * prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(
            slice(_bs[0] if _bs[0] > 0 else None, -_bs[1] if _bs[1] > 0 else None) for _bs in b
        )
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh


def non_maximum_suppression_sparse(dist, prob, points, nms_thresh=0.5):
    """Non-Maximum-Supression of 2D polygons from a list of dists, probs (scores), and points.

    Retains only polyhedra whose overlap is smaller than nms_thresh

    dist.shape = (n_polys, n_rays)
    prob.shape = (n_polys,)
    points.shape = (n_polys,2)

    returns the retained instances

    (pointsi, probi, disti, indsi)

    with
    pointsi = points[indsi] ...

    """
    dist = np.asarray(dist)
    prob = np.asarray(prob)
    points = np.asarray(points)

    assert dist.ndim == 2, "dist must have 2 dimensions"
    assert prob.ndim == 1, "prob must be a 1-dimensional array"
    assert points.ndim == 2, "points must have 2 dimensions"
    assert points.shape[-1] == 2, "points must have 2 columns"
    assert len(prob) == len(dist) == len(points), "prob, dist, and points must have the same length"

    inds_original = np.arange(len(prob))
    _sorted = np.argsort(prob)[::-1]
    probi = prob[_sorted]
    disti = dist[_sorted]
    pointsi = points[_sorted]
    inds_original = inds_original[_sorted]

    inds = p_non_max_suppression_inds(disti, pointsi, threshold=nms_thresh)

    return pointsi[inds], probi[inds], disti[inds], inds_original[inds]


def p_non_max_suppression_inds(dist_array, points_array, threshold=0.5):
    """Perform non-maximum suppression on polygons defined by rays from center points.

    Args:
        dist_array: numpy array of shape (n_polys, n_rays) containing ray distances
        points_array: numpy array of shape (n_polys, 2) containing center points (y,x)
        threshold: float, IoU threshold for suppression

    Returns:
        numpy array of shape (n_polys,) with False indicating suppressed polygons
    """
    assert dist_array.ndim == 2
    assert points_array.ndim == 2

    n_polys, n_rays = dist_array.shape
    assert points_array.shape[0] == n_polys

    angle_pi = 2 * math.pi / n_rays

    sin_angles = np.ones(n_rays)
    cos_angles = np.ones(n_rays)

    for k in range(n_rays):
        sin_angles[k] = math.sin(angle_pi * k)
        cos_angles[k] = math.cos(angle_pi * k)

    # coordinates of all polygons
    p_coords = np.zeros((n_polys, n_rays, 2))
    p_coords[:, :, 0] = np.expand_dims(points_array[:, 1], axis=0).T + dist_array * cos_angles
    p_coords[:, :, 1] = np.expand_dims(points_array[:, 0], axis=0).T + dist_array * sin_angles

    print(f"NMS: n_polys    = {n_polys}")

    polygons = [Polygon(coords) for coords in p_coords]
    str_tree = STRtree(polygons)

    # Initialize arrays
    suppressed = np.zeros(n_polys, dtype=bool)

    # radius of polygon
    radius_outer = np.max(dist_array, axis=1)

    # max of all radii
    max_dist = np.max(radius_outer)

    count_suppressed = 0

    # Main suppression loop
    for i in range(n_polys - 1):
        if not suppressed[i]:
            neighbors = str_tree.query(
                polygons[i], predicate="intersects", distance=max_dist + radius_outer[i]
            )
            neighbors = [j for j in neighbors if j > i]

            # Check each neighbor
            for j in neighbors:
                if not suppressed[j]:
                    # compute ooverlap
                    overlap = polygons[i].intersection(polygons[j]).area / min(
                        polygons[i].area, polygons[j].area
                    )

                    if overlap > threshold:
                        suppressed[j] = True
                        count_suppressed += 1

    print(
        f"NMS: Suppressed polygons: {count_suppressed:8d} / {n_polys}"
        f"  ({100.0 * count_suppressed / n_polys:.2f} %)"
    )

    return ~suppressed
