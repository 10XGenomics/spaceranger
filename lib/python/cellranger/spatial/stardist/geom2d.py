#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""2D geometry utilities."""

import numpy as np
from skimage.draw import polygon  # pylint:disable=no-name-in-module

from cellranger.spatial.stardist.matching import _check_label_array


def dist_to_coord(dist, points, scale_dist=(1, 1)):
    """Convert from polar to cartesian coordinates for a list of distances and center points.

    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)
    len(scale_dist) = 2
    return coord.shape = (n_polys,2,n_rays)
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    assert dist.ndim == 2, "dist must have 2 dimensions"
    assert points.ndim == 2, "points must have 2 dimensions"
    assert len(dist) == len(points), "dist and points must have the same length"
    assert points.shape[1] == 2, "points must have 2 columns"
    assert len(scale_dist) == 2, "scale_dist must have a length of 2"
    n_rays = dist.shape[1]
    phis = ray_angles(n_rays)
    coord = (dist[:, np.newaxis] * np.array([np.sin(phis), np.cos(phis)])).astype(np.float32)
    coord *= np.asarray(scale_dist).reshape(1, 2, 1)
    coord += points[..., np.newaxis]
    return coord


def polygons_to_label_coord(coord, shape, labels=None):
    """Renders polygons to image of given shape.

    coord.shape   = (n_polys, n_rays)
    """
    coord = np.asarray(coord)
    if labels is None:
        labels = np.arange(len(coord))

    _check_label_array(labels, "labels")
    assert coord.ndim == 3, "coord must have 3 dimensions"
    assert coord.shape[1] == 2, "coord must have 2 columns"
    assert len(coord) == len(labels), "coord and labels must have the same length"

    lbl = np.zeros(shape, np.int32)

    for i, c in zip(labels, coord):
        rr, cc = polygon(*c, shape)
        lbl[rr, cc] = i + 1

    return lbl


def polygons_to_label(dist, points, shape, prob=None, thr=-np.inf, scale_dist=(1, 1)):
    """Converts distances and center points to label image.

    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)

    label ids will be consecutive and adhere to the order given
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    prob = np.inf * np.ones(len(points)) if prob is None else np.asarray(prob)

    assert dist.ndim == 2, "dist must have 2 dimensions"
    assert points.ndim == 2, "points must have 2 dimensions"
    assert len(dist) == len(points), "dist and points must have the same length"
    assert len(points) == len(prob), "points and prob must have the same length"
    assert points.shape[1] == 2, "points must have 2 columns"
    assert prob.ndim == 1, "prob must be a 1-dimensional array"

    ind = prob > thr
    points = points[ind]
    dist = dist[ind]
    prob = prob[ind]

    ind = np.argsort(prob, kind="stable")
    points = points[ind]
    dist = dist[ind]

    coord = dist_to_coord(dist, points, scale_dist=scale_dist)

    return polygons_to_label_coord(coord, shape=shape, labels=ind)


def ray_angles(n_rays=32):
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
