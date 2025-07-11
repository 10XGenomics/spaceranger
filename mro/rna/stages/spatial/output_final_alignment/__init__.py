#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Outputs fiducial and tissue transforms."""
import json
from dataclasses import asdict

import martian
import numpy as np

from cellranger.feature.utils import write_json_from_dict
from cellranger.spatial.bounding_box import BoundingBox
from cellranger.spatial.data_utils import parse_slide_sample_area_id
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.transform import (
    normalize_perspective_transform,
    scale_matrix,
    transform_pts_2d,
)

__MRO__ = """
stage OUTPUT_FINAL_ALIGNMENT(
    in  string slide_serial_capture_area,
    in  file[] cytassist_image_paths,
    in  file[] tissue_image_paths,
    in  json   registered_spots_data_json,
    in  json   final_transform_json,
    in  json   transform_matrix,
    in  json   registered_selected_spots_json,
    in  json   scalefactors_json,
    out json   fiducial_bounding_box_on_tissue_image,
    src py     "stages/spatial/output_final_alignment",
) using (
    volatile = strict,
)
"""


def main(args, outs):

    # Add tissue registration and slide info to alignment file and save
    if args.registered_selected_spots_json:
        spots_data = LoupeParser(args.registered_selected_spots_json)
    elif args.registered_spots_data_json:
        spots_data = LoupeParser(args.registered_spots_data_json)
    else:
        martian.clear(outs)
        return

    if not spots_data.has_spot_transform() and args.transform_matrix:
        with open(args.transform_matrix) as f:
            transform = json.load(f)
        spots_data.set_spot_transform(transform)
    scale = 1
    if args.scalefactors_json:
        with open(args.scalefactors_json) as f:
            scalefactors = json.load(f)
            if "regist_target_img_scalef" in scalefactors:
                scale = scalefactors["regist_target_img_scalef"]

    if args.tissue_image_paths and args.final_transform_json:
        with open(args.final_transform_json) as f:
            transform_mat = json.load(f)["tissue_transform"]
        loupe_transform = normalize_perspective_transform(
            np.linalg.inv(transform_mat) @ scale_matrix(scale)
        )
        spots_data.update_tissue_transform(loupe_transform.tolist(), args.tissue_image_paths[0])

        # Obtain bounding box of fiducial area in tissue image space
        target_fid_xy = transform_pts_2d(
            spots_data.get_fiducials_imgxy(), scale_matrix(1.0 / scale) @ transform_mat
        )
        target_fid_bbox = BoundingBox.new(
            minr=np.min(target_fid_xy[:, 1]),
            minc=np.min(target_fid_xy[:, 0]),
            maxr=np.max(target_fid_xy[:, 1]),
            maxc=np.max(target_fid_xy[:, 0]),
        )
        write_json_from_dict(asdict(target_fid_bbox), outs.fiducial_bounding_box_on_tissue_image)
    else:
        outs.fiducial_bounding_box_on_tissue_image = None

    if args.cytassist_image_paths:
        spots_data.update_checksum(args.cytassist_image_paths[0])
    if args.slide_serial_capture_area:
        sample_id, area_id = parse_slide_sample_area_id(args.slide_serial_capture_area)
        spots_data.set_serial_number(sample_id)
        spots_data.set_area(area_id)
    else:
        spots_data.set_serial_number("")
        spots_data.set_area("")
