#!/usr/bin/env python3
#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#

"""Stage to generate the final output of oligos and barcodes."""

from __future__ import annotations

import json
import os

import numpy as np

import cellranger.barcodes.utils as bc_utils
import cellranger.spatial.spot_barcode_utils as sbu
import cellranger.spatial.utils as spatial_utils
import tenkit.safe_json as tk_safe_json
from cellranger.fast_utils import (  # pylint: disable=no-name-in-module,unused-import
    SquareBinIndex,
)
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.slide_design_o3 import VisiumHdSlideWrapper
from cellranger.spatial.transform import (
    normalize_perspective_transform,
    scale_matrix,
    transform_pts_2d,
)

__MRO__ = """
stage OUTPUT_SPOTS_BARCODES(
    in  json   registered_selected_spots_json,
    in  json   tissue_transform_json,
    in  json   scalefactors,
    in  string barcode_whitelist,
    in  string visium_hd_slide_name,
    in  png    tissue_lowres_image,
    out txt    final_spot_position_list,
    out txt    final_fiducial_position_list,
    out csv    tissue_positions,
    out json   barcodes_under_tissue,
    out float  fraction_under_tissue,
    out json   tissue_final_transform_json,
    out json   scalefactors,
    out json   fraction_bc_outside_image,
    src py     "stages/spatial/output_spots_barcodes",
) split (
) using (
    volatile = strict,
)
"""


def split(args):
    mem_gb = max(
        8, 1.0 + LoupeParser.estimate_mem_gb_from_json_file(args.registered_selected_spots_json)
    )

    return {
        "chunks": [],
        "join": {
            "__mem_gb": mem_gb,
        },
    }


# pylint: disable=too-many-locals
def join(args, outs, _chunk_defs, _chunk_outs):
    with open(args.scalefactors) as f:
        scale_dict = json.load(f)

    process_img_scalef = scale_dict.pop("process_img_scalef")

    if args.tissue_transform_json is not None and os.path.isfile(args.tissue_transform_json):
        microscope_img_scale = scale_dict["regist_target_img_scalef"]
        with open(args.tissue_transform_json) as f:
            regist_mat = json.load(f)["tissue_transform"]

        transform = normalize_perspective_transform(
            scale_matrix(1 / microscope_img_scale) @ regist_mat
        )
        # only save the transform if tissue registration is done
        with open(outs.tissue_final_transform_json, "w") as f:
            tk_safe_json.dump_numpy(transform, f)
    else:
        outs.tissue_final_transform_json = None
        transform = scale_matrix(1 / process_img_scalef)

    scale = np.sqrt(np.abs(transform[0, 0] * transform[1, 1] - transform[0, 1] * transform[1, 0]))
    spots_data = LoupeParser(args.registered_selected_spots_json)

    # update scalefactor_json with spot diameter
    scale_dict["fiducial_diameter_fullres"] = spots_data.get_fiducials_diameter() * scale
    scale_dict["spot_diameter_fullres"] = spots_data.get_oligos_diameter() * scale

    with open(outs.scalefactors, "w") as f:
        json.dump(scale_dict, f)

    oligo_under_tissue = spots_data.tissue_oligos_flags()

    transf_fid_xy = transform_pts_2d(spots_data.get_fiducials_imgxy(), transform)
    if transf_fid_xy.size == 0:
        final_fid_list = []
    else:
        fid_coords = np.round(np.flip(transf_fid_xy, axis=1)).astype(int)
        final_fid_list = np.concatenate((spots_data.get_fiducials_rowcol(), fid_coords), axis=1)
    spatial_utils.write_to_list(outs.final_fiducial_position_list, final_fid_list)

    if args.barcode_whitelist:
        transf_oligos_xy = transform_pts_2d(spots_data.get_oligos_imgxy(), transform)
        # from (x, y) to (row, col) to save as list
        oligos_coords = np.flip(transf_oligos_xy, axis=1)
        final_oligo_list = np.concatenate((spots_data.get_oligos_rowcol(), oligos_coords), axis=1)
        # output result to files
        spatial_utils.write_to_list(outs.final_spot_position_list, final_oligo_list)

        bc_coord_file = bc_utils.get_barcode_whitelist_path(args.barcode_whitelist + "_coordinates")
        barcodes = sbu.read_barcode_coordinates(bc_coord_file)

        tissue_barcodes = sbu.save_tissue_position_list(
            final_oligo_list,
            oligo_under_tissue,
            barcodes,
            outs.tissue_positions,
            gemgroup=sbu.GEM_GROUP,
        )
        outs.fraction_under_tissue = len(tissue_barcodes) / len(barcodes)
        with open(outs.barcodes_under_tissue, "w") as f:
            json.dump(tissue_barcodes, f)

        # Check if over 90% of the spots are outside the microscope image which indicates possible tissue registration error.
        fraction_bc_outside_image = sbu.spots_outside_image(
            tissue_positions=outs.tissue_positions,
            tissue_lowres_image=args.tissue_lowres_image,
            scale_factors=args.scalefactors,
        )
        with open(outs.fraction_bc_outside_image, "w") as f:
            json.dump(fraction_bc_outside_image, f)

    elif args.visium_hd_slide_name:
        outs.final_spot_position_list = None
        outs.tissue_positions = None
        outs.fraction_bc_outside_image = None
        slide = VisiumHdSlideWrapper(slide_name=args.visium_hd_slide_name)
        tissue_barcodes = [
            f"{SquareBinIndex(row=spot.grid_index.row, col=spot.grid_index.col, size_um=int(slide.spot_pitch()))}-{sbu.GEM_GROUP}"
            for (in_tissue, spot) in zip(oligo_under_tissue, slide.spots())
            if in_tissue
        ]
        outs.fraction_under_tissue = len(tissue_barcodes) / slide.num_spots()

        with open(outs.barcodes_under_tissue, "w") as f:
            json.dump(tissue_barcodes, f)
