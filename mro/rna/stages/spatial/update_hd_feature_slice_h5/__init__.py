# Copyright (c) 2021 10x Genomics, Inc. All rights reserved.
"""Update feature slice H5 for HD.

- Add the mask of filtered barcodes
"""

from __future__ import annotations

import os
import shutil

import cellranger.matrix as cr_matrix
from cellranger.spatial.hd_feature_slice import HdFeatureSliceIo, bin_size_um_from_bin_name

__MRO__ = """
stage UPDATE_HD_FEATURE_SLICE_H5(
    in  h5        filtered_matrix_h5,
    in  h5        hd_feature_slice_h5_in,
    in  map<path> binned_analysis,
    in  png       cytassist_image_on_spots,
    in  png       microscope_image_on_spots,
    out h5        hd_feature_slice_h5_out    "HD Feature Slice"  "hd_feature_slice.h5",
    src py        "stages/spatial/update_hd_feature_slice_h5",
) split (
)
"""


def split(args):
    return {
        "chunks": [],
        "join": {
            "__mem_gb": cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(
                args.filtered_matrix_h5, scale=6.5
            )
            + 2.5,
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):
    shutil.copyfile(
        args.hd_feature_slice_h5_in, outs.hd_feature_slice_h5_out, follow_symlinks=False
    )
    feature_slice_io = HdFeatureSliceIo(h5_path=outs.hd_feature_slice_h5_out, open_mode="a")
    slide = feature_slice_io.slide()

    # Write mask for the finest bin level
    feature_slice_io.write_filtered_mask(
        slide=slide, filtered_matrix=cr_matrix.CountMatrix.load_h5_file(args.filtered_matrix_h5)
    )

    if args.microscope_image_on_spots:
        feature_slice_io.write_microscope_image_on_spots(args.microscope_image_on_spots)
    if args.cytassist_image_on_spots:
        feature_slice_io.write_cytassist_image_on_spots(args.cytassist_image_on_spots)

    filtered_bcs = cr_matrix.CountMatrix.load_bcs_from_h5_file(args.filtered_matrix_h5)
    # Secondary analysis slices
    if args.binned_analysis:
        for bin_name, analysis_folder in args.binned_analysis.items():
            # Write mask for all bin levels apart from the finest bin level
            if bin_size_um_from_bin_name(bin_name=bin_name) != feature_slice_io.metadata.spot_pitch:
                feature_slice_io.write_binned_filtered_mask(
                    bin_name=bin_name, filtered_unbinned_bcs=filtered_bcs
                )
            if analysis_folder is not None:
                analysis_h5 = os.path.join(analysis_folder, "analysis.h5")
                if os.path.exists(analysis_h5):
                    feature_slice_io.write_secondary_analysis(bin_name, analysis_h5)
