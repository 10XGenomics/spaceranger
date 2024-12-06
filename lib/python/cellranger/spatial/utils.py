#!/usr/bin/env python
#
# Copyright (c) 2019 10X Genomics, Inc. All rights reserved.
#

# Do not add new things to this module.
# Instead, either find or create a module with a name that better describes
# the functionality implemented by the methods or classes you want to add.
import json

import numpy as np


def update_scalefactors_for_diameters(in_json_path, fid_dia, spot_dia, out_json_path):
    """Serialize (optional) fiducial diameter and spot diameter to scalefactors JSON."""
    scalefactors = json.load(open(in_json_path))

    if fid_dia is not None:
        scalefactors["fiducial_diameter_fullres"] = fid_dia
    if spot_dia is not None:
        scalefactors["spot_diameter_fullres"] = spot_dia

    json.dump(scalefactors, open(out_json_path, "w"))


def transform_oligo_locations(
    spots,
    destination_points,
    source_points,
    inverse_rotation_mtx,
    inverse_t_matrix,
    transf_mtx,
    scale,
):
    """Given oligo locations from a GPR file, transform their x/y coordinates using the outputs of fiducial alignment.

    Fiducial alignment might have needed to rotate or mirror the fiducial pattern; specified via transf_mtx
    """
    oligo_coordinates = np.array(
        [[oligo["row"], oligo["col"], oligo["x"], oligo["y"]] for oligo in spots["oligo"]]
    )
    spots_xy = oligo_coordinates[:, 2:4].astype("double")
    spots_centered = spots_xy - np.median(destination_points, 0)
    spots_centered = np.transpose(np.dot(transf_mtx, np.transpose(spots_centered)))

    transformed_xy = np.dot(
        spots_centered + np.tile(inverse_t_matrix, (np.shape(spots_centered)[0], 1)),
        inverse_rotation_mtx,
    )
    uncentered_xy = transformed_xy + np.median(source_points, 0)
    uncentered_xy = np.vstack([uncentered_xy[:, 1], uncentered_xy[:, 0]]).T
    scaled_xy = uncentered_xy / scale
    transformed_coordinates = np.hstack([oligo_coordinates[:, 0:2], scaled_xy]).astype(int)

    return transformed_coordinates


def frac_bad_transformed_coordinates(transformed_coordinates, iscale, shape):
    """Count fraction of transformed coordinates that are outside of the image."""
    if len(transformed_coordinates) == 0:
        return 1.0

    bad = 0
    for _, _, y, x in transformed_coordinates:
        x *= iscale
        y *= iscale
        if x < 0 or x > shape[1] - 1 or y < 0 or y > shape[0] - 1:
            bad += 1

    return float(bad) / len(transformed_coordinates)


def write_to_list(lst, coordinates):
    """Write given coordinates to a txt file."""
    with open(lst, "w") as fd:
        for spot in coordinates:
            fd.write(",".join(map(str, spot)))
            fd.write("\n")
